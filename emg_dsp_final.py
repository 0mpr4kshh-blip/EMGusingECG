"""
EMG muscle-activity detector (AD8232 + Arduino Uno) - DSP written from scratch.

Hand-implemented (the actual DSP, no library calls in the processing path):
  * Biquad IIR filters, Direct Form II Transposed (sample-by-sample, causal)
  * RBJ-cookbook coefficient design: high-pass, low-pass, notch (bilinear transform)
  * 4th-order Butterworth high-pass / low-pass as two cascaded biquads
  * Recursive moving-RMS envelope (O(1) per sample)
  * Radix-2 FFT, Hann window, Welch PSD
  * Run-length burst detection, robust (median + k*MAD) threshold, hysteresis detector
  * Time/frequency features: MAV, RMS, WL, ZC, MDF, MNF

SciPy is used ONLY in validate() to prove the hand-written code is correct.

Usage:
  python3 emg_dsp.py record /dev/ttyACM0 15      # save raw ADC to emg_raw.csv
  python3 emg_dsp.py analyze emg_raw.csv         # offline analysis + figures
  python3 emg_dsp.py live /dev/ttyACM0 --plot    # real-time ACTIVE / REST
  python3 emg_dsp.py live emg_raw.csv            # replay a CSV in real time (no hardware)
  python3 emg_dsp.py synth                       # make emg_synth.csv for testing
"""
import sys, csv, time, argparse
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

# ---------------- settings ----------------
FS = 1000          # Hz, must match Arduino
BAUD = 115200      # must match Arduino
VREF = 3.3         # 3.3 if AREF is wired to 3.3 V (and USE_AREF = true in sketch)
MAINS = 50         # Hz (India)
BAND = (10, 45)    # AD8232 module passes roughly 0.5-40 Hz
NOTCH_Q = 30
SETTLE_S = 1.0     # discard filter start-up transient
ENV_MS = 100       # RMS envelope window
# 4th-order Butterworth = two biquads with these Q values: Q_k = 1/(2cos(k*pi/8)), k=1,3
BUTTER4_Q = tuple(1 / (2 * np.cos(k * np.pi / 8)) for k in (1, 3))   # 0.5412, 1.3066
# ------------------------------------------


# =====================================================================
# 1. Core DSP building blocks (hand-written)
# =====================================================================
class Biquad:
    """y[n] = b0 x[n] + z1 ; z1 = b1 x[n] - a1 y[n] + z2 ; z2 = b2 x[n] - a2 y[n]
    (Direct Form II Transposed; coefficients already normalised so a0 = 1)."""

    def __init__(self, b0, b1, b2, a1, a2):
        self.b = (b0, b1, b2)
        self.a = (1.0, a1, a2)
        self.z1 = self.z2 = 0.0

    def step(self, x):
        b0, b1, b2 = self.b
        _, a1, a2 = self.a
        y = b0 * x + self.z1
        self.z1 = b1 * x - a1 * y + self.z2
        self.z2 = b2 * x - a2 * y
        return y

    def process(self, x):
        return np.array([self.step(s) for s in x])


def notch_coeffs(f0, Q, fs):
    """Bilinear transform of the analog notch (s^2+w0^2)/(s^2+(w0/Q)s+w0^2), unity DC gain.
    beta = tan(BW/2), BW = w0/Q;  g = 1/(1+beta)
    H(z) = g(1 - 2cos(w0) z^-1 + z^-2) / (1 - 2g cos(w0) z^-1 + (2g-1) z^-2)"""
    w0 = 2 * np.pi * f0 / fs
    beta = np.tan(w0 / Q / 2)
    g = 1 / (1 + beta)
    c = np.cos(w0)
    return (g, -2 * g * c, g, -2 * g * c, 2 * g - 1)


def highpass_coeffs(fc, Q, fs):
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * Q)
    c = np.cos(w0)
    a0 = 1 + alpha
    return ((1 + c) / 2 / a0, -(1 + c) / a0, (1 + c) / 2 / a0,
            -2 * c / a0, (1 - alpha) / a0)


def lowpass_coeffs(fc, Q, fs):
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * Q)
    c = np.cos(w0)
    a0 = 1 + alpha
    return ((1 - c) / 2 / a0, (1 - c) / a0, (1 - c) / 2 / a0,
            -2 * c / a0, (1 - alpha) / a0)


def build_chain():
    """HPF 10 Hz (4th order) -> LPF 45 Hz (4th order) -> 50 Hz notch. 5 biquads."""
    chain = [Biquad(*highpass_coeffs(BAND[0], q, FS)) for q in BUTTER4_Q]
    chain += [Biquad(*lowpass_coeffs(BAND[1], q, FS)) for q in BUTTER4_Q]
    chain.append(Biquad(*notch_coeffs(MAINS, NOTCH_Q, FS)))
    return chain


def my_preprocess(x):
    y = np.asarray(x, float)
    for stage in build_chain():
        y = stage.process(y)
    return y


class MovingRMS:
    """Causal moving RMS using a running sum of squares (O(1) per sample)."""

    def __init__(self, n):
        self.n = n
        self.buf = deque([0.0] * n, maxlen=n)
        self.acc = 0.0

    def step(self, x):
        sq = x * x
        self.acc += sq - self.buf[0]   # add newest, drop oldest
        self.buf.append(sq)
        return np.sqrt(max(self.acc, 0.0) / self.n)

    def process(self, x):
        return np.array([self.step(s) for s in x])


def my_fft(x):
    """Recursive radix-2 decimation-in-time FFT (length must be a power of 2)."""
    x = np.asarray(x, complex)
    N = len(x)
    if N == 1:
        return x
    E = my_fft(x[0::2])
    O = my_fft(x[1::2])
    tw = np.exp(-2j * np.pi * np.arange(N // 2) / N) * O
    return np.concatenate([E + tw, E - tw])


def my_psd(x, nseg=256, overlap=0.5):
    """Welch PSD (V^2/Hz): Hann window, 50 % overlap, hand-written FFT, averaged."""
    x = np.asarray(x, float)
    if len(x) < nseg:
        nseg = 2 ** int(np.log2(len(x)))
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nseg) / nseg)   # periodic Hann
    U = np.sum(w ** 2)
    step = max(1, int(nseg * (1 - overlap)))
    acc, k = np.zeros(nseg // 2 + 1), 0
    for i in range(0, len(x) - nseg + 1, step):
        seg = x[i:i + nseg]
        X = my_fft((seg - seg.mean()) * w)[: nseg // 2 + 1]
        acc += np.abs(X) ** 2 / (FS * U)
        k += 1
    P = acc / k
    P[1:-1] *= 2                                                  # one-sided
    return np.arange(nseg // 2 + 1) * FS / nseg, P


def find_runs(mask):
    """Return [(start, end_exclusive), ...] of consecutive True samples."""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


# =====================================================================
# 2. Detection and features
# =====================================================================
def adc_to_volts(x):
    x = np.asarray(x, float)
    nans = np.isnan(x)
    if nans.all():
        raise SystemExit("All samples are lead-off ('L'). Check electrode connections.")
    if nans.any():
        x[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), x[~nans])
    v = x * VREF / 1023.0
    return v - v.mean()


def robust_threshold(env, k=4.0):
    med = np.median(env)
    mad = np.median(np.abs(env - med)) * 1.4826
    return max(med + k * mad, 1.5 * med), med


def detect_activity(env, k=4.0, min_ms=50):
    """Offline: threshold = median + k*MAD of the whole envelope; drop bursts < min_ms."""
    thr, _ = robust_threshold(env, k)
    active = env > thr
    min_n = int(min_ms * FS / 1000)
    for s, e in find_runs(active):
        if e - s < min_n:
            active[s:e] = False
    return active, thr


class ActivityDetector:
    """Streaming detector with hysteresis and debounce."""

    def __init__(self, thr_on, thr_off, min_on_ms=50, min_off_ms=100):
        self.thr_on, self.thr_off = thr_on, thr_off
        self.min_on = int(FS * min_on_ms / 1000)
        self.min_off = int(FS * min_off_ms / 1000)
        self.active = False
        self.on_cnt = self.off_cnt = self.events = 0

    def step(self, e):
        if not self.active:
            self.on_cnt = self.on_cnt + 1 if e > self.thr_on else 0
            if self.on_cnt >= self.min_on:
                self.active, self.off_cnt = True, 0
                self.events += 1
        else:
            self.off_cnt = self.off_cnt + 1 if e < self.thr_off else 0
            if self.off_cnt >= self.min_off:
                self.active, self.on_cnt = False, 0
        return self.active


def features(v, active):
    seg = v[active]
    if len(seg) < 50:
        return {}
    mav = np.mean(np.abs(seg))
    rms = np.sqrt(np.mean(seg ** 2))
    wl = np.sum(np.abs(np.diff(seg)))
    zc = int(np.sum(np.diff(np.sign(seg)) != 0))
    f, p = my_psd(seg)
    cum = np.cumsum(p)
    mdf = f[np.searchsorted(cum, cum[-1] / 2)]
    mnf = np.sum(f * p) / np.sum(p)
    return dict(MAV=mav, RMS=rms, WL=wl, ZC=zc, MDF=mdf, MNF=mnf)


def decide(env, active, min_fraction=0.02, min_contrast=1.5):
    runs = find_runs(active)
    frac = active.mean()
    contrast = np.percentile(env, 98) / max(np.median(env), 1e-12)
    is_active = (len(runs) >= 1) and (frac >= min_fraction) and (contrast >= min_contrast)
    return dict(is_active=is_active, n_bursts=len(runs), active_pct=frac * 100,
                contrast=contrast, bursts=[(s / FS, (e - 1) / FS) for s, e in runs])


# =====================================================================
# 3. Raw vs processed: numerical performance measures
# =====================================================================
def compare(raw, proc, active):
    f, Pr = my_psd(raw)
    _, Pp = my_psd(proc)
    df = f[1] - f[0]

    def bp(P, lo, hi):
        return P[(f >= lo) & (f <= hi)].sum() * df

    def snr(x):
        rest = ~active
        if not active.any() or not rest.any():
            return float("nan")
        return 10 * np.log10(np.mean(x[active] ** 2) / np.mean(x[rest] ** 2))

    res = dict(
        mains_red_db=10 * np.log10(bp(Pr, MAINS - 2, MAINS + 2) / bp(Pp, MAINS - 2, MAINS + 2)),
        inband_raw=bp(Pr, *BAND) / bp(Pr, 0, FS / 2),
        inband_proc=bp(Pp, *BAND) / bp(Pp, 0, FS / 2),
        rms_raw=raw.std(), rms_proc=proc.std(),
        snr_raw=snr(raw), snr_proc=snr(proc),
        rest_rms_raw=raw[~active].std() if (~active).any() else float("nan"),
        rest_rms_proc=proc[~active].std() if (~active).any() else float("nan"),
    )
    res["snr_gain"] = res["snr_proc"] - res["snr_raw"]
    print("Raw vs processed (numerical):")
    print(f"  Mains {MAINS} Hz (+-2) power reduction : {res['mains_red_db']:.1f} dB")
    print(f"  In-band {BAND} Hz power fraction : raw {res['inband_raw']:.1%} -> "
          f"processed {res['inband_proc']:.1%}")
    print(f"  Total RMS                          : raw {res['rms_raw']:.4f} V -> "
          f"processed {res['rms_proc']:.4f} V")
    print(f"  Rest-period noise RMS              : raw {res['rest_rms_raw']:.4f} V -> "
          f"processed {res['rest_rms_proc']:.4f} V")
    print(f"  Active/rest SNR                    : raw {res['snr_raw']:.1f} dB -> "
          f"processed {res['snr_proc']:.1f} dB  (gain {res['snr_gain']:+.1f} dB)")
    return res


# =====================================================================
# 4. Validation against SciPy (the ONLY place SciPy is used)
# =====================================================================
def validate(raw_v, plot=True):
    from scipy import signal
    print("Validation of hand-written DSP against SciPy:")

    # a) time-domain: same causal cascade in SciPy
    sos = np.vstack([signal.butter(4, BAND[0], "high", fs=FS, output="sos"),
                     signal.butter(4, BAND[1], "low", fs=FS, output="sos")])
    ref = signal.sosfilt(sos, raw_v)
    bn, an = signal.iirnotch(MAINS, NOTCH_Q, fs=FS)
    ref = signal.lfilter(bn, an, ref)
    mine = my_preprocess(raw_v)
    err = np.max(np.abs(mine - ref))
    print(f"  Filter chain max |mine - scipy|  : {err:.2e} V  "
          f"({err / np.std(ref):.2e} of signal std)")

    # b) frequency response
    freqs = np.linspace(0.5, 200, 2000)
    w = 2 * np.pi * freqs / FS
    H = np.ones_like(freqs, complex)
    for s in build_chain():
        H *= signal.freqz(s.b, s.a, worN=w)[1]
    _, Hs = signal.sosfreqz(sos, worN=w)
    Hs = Hs * signal.freqz(bn, an, worN=w)[1]
    d_db = np.max(np.abs(20 * np.log10(np.abs(H) + 1e-12) - 20 * np.log10(np.abs(Hs) + 1e-12)))
    print(f"  Frequency response max diff      : {d_db:.2e} dB")

    # c) PSD
    f1, P1 = my_psd(mine)
    f2, P2 = signal.welch(mine, FS, nperseg=256)
    rel = np.max(np.abs(P1 - P2) / np.maximum(P2, 1e-18))
    print(f"  Welch PSD max relative error     : {rel:.2e}")

    # d) FFT
    x = np.random.default_rng(1).standard_normal(256)
    ferr = np.max(np.abs(my_fft(x) - np.fft.fft(x)))
    print(f"  FFT max |mine - numpy|           : {ferr:.2e}")

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(freqs, 20 * np.log10(np.abs(H) + 1e-12), label="hand-written")
        ax[0].plot(freqs, 20 * np.log10(np.abs(Hs) + 1e-12), "--", label="SciPy")
        ax[0].set_ylim(-80, 5); ax[0].set_xlabel("Hz"); ax[0].set_ylabel("dB")
        ax[0].set_title("Filter chain magnitude response"); ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].semilogy(f1, P1, label="hand-written Welch")
        ax[1].semilogy(f2, P2, "--", label="scipy.signal.welch")
        ax[1].set_xlim(0, 100); ax[1].set_xlabel("Hz"); ax[1].set_title("PSD of processed EMG")
        ax[1].legend(); ax[1].grid(alpha=.3)
        plt.tight_layout(); fig.savefig("emg_validation.png", dpi=150)
    return dict(filter_err=err, resp_db=d_db, psd_rel=rel, fft_err=ferr)


# =====================================================================
# 5. Offline analysis
# =====================================================================
def analyze(path, plot=True):
    adc = np.genfromtxt(path, delimiter=",", skip_header=1)
    raw_v = adc_to_volts(adc)
    proc_full = my_preprocess(raw_v)
    skip = int(SETTLE_S * FS)
    raw, v = raw_v[skip:], proc_full[skip:]
    env = MovingRMS(int(FS * ENV_MS / 1000)).process(v)
    print(f"{len(raw_v)} samples ({len(raw_v) / FS:.1f} s); first {SETTLE_S:.1f} s "
          f"discarded (filter settling)")

    n_sec = len(env) // FS
    if n_sec >= 2:
        sec = env[: n_sec * FS].reshape(-1, FS).mean(axis=1)
        print("Envelope per second:", np.round(sec, 4))
        print(f"Max/min second ratio: {sec.max() / max(sec.min(), 1e-9):.2f}")

    active, thr = detect_activity(env)
    feats = features(v, active)
    res = decide(env, active)
    verdict = "MUSCLE ACTIVE" if res["is_active"] else "MUSCLE NOT ACTIVE (rest)"
    print("=" * 50)
    print(f"RESULT: {verdict}")
    print(f"  Bursts detected : {res['n_bursts']}")
    print(f"  Active time     : {res['active_pct']:.1f} %")
    print(f"  Contrast        : {res['contrast']:.2f}x (needs >= 1.5)")
    for i, (s, e) in enumerate(res["bursts"], 1):
        print(f"  Burst {i}: {s + SETTLE_S:.2f} s -> {e + SETTLE_S:.2f} s ({e - s:.2f} s)")
    print("=" * 50)
    if feats:
        print("Features of active segments:")
        for k_, val in feats.items():
            print(f"  {k_}: {val:.4f}")
    print()
    compare(raw, v, active)
    print()
    validate(raw_v, plot)

    if plot:
        t = (np.arange(len(v)) + skip) / FS
        fig, ax = plt.subplots(4, 1, figsize=(11, 11))
        fig.suptitle(verdict, fontsize=16, fontweight="bold",
                     color="green" if res["is_active"] else "red")
        ax[0].plot(t, raw, lw=0.5, color="gray"); ax[0].set_title("Raw EMG, mean removed (V)")
        ax[1].plot(t, v, lw=0.5); ax[1].set_title("Filtered EMG (V): HPF 10 Hz, LPF 45 Hz, 50 Hz notch")
        ax[2].plot(t, env); ax[2].axhline(thr, color="r", ls="--", label="threshold")
        ax[2].fill_between(t, 0, env.max(), where=active, alpha=0.2, color="g")
        ax[2].set_title("RMS envelope + detected activity"); ax[2].legend()
        f1, P1 = my_psd(raw); f2, P2 = my_psd(v)
        ax[3].semilogy(f1, P1, color="gray", label="raw")
        ax[3].semilogy(f2, P2, label="processed")
        ax[3].axvline(MAINS, color="r", ls=":", label=f"{MAINS} Hz mains")
        ax[3].set_xlim(0, 100); ax[3].set_xlabel("Hz"); ax[3].legend()
        ax[3].set_title("Power spectral density (hand-written Welch)")
        plt.tight_layout(); fig.savefig("emg_analysis.png", dpi=150)
        print("\nSaved emg_analysis.png and emg_validation.png")
        plt.show()


# =====================================================================
# 6. Real-time (sample-by-sample) mode
# =====================================================================
def sample_source(src, fast=False):
    """Yield ADC ints (nan for lead-off) from a serial port or a CSV replay."""
    if src.lower().endswith(".csv"):
        data = np.genfromtxt(src, delimiter=",", skip_header=1)
        t0 = time.time()
        for i, d in enumerate(data):
            if not fast:
                delay = t0 + i / FS - time.time()
                if delay > 0.002:
                    time.sleep(delay)
            yield d
    else:
        import serial
        ser = serial.Serial(src, BAUD, timeout=1)
        time.sleep(2)
        ser.reset_input_buffer()
        try:
            while True:
                line = ser.readline().decode(errors="ignore").strip()
                if line == "L":
                    yield np.nan
                elif line.isdigit():
                    yield int(line)
        finally:
            ser.close()


def live(src, seconds=None, fast=False, plot=False, k=4.0, cal_s=3.0, log="live_log.csv"):
    chain = build_chain()
    rms = MovingRMS(int(FS * ENV_MS / 1000))
    OFFSET_N, settle_n, cal_n = 500, int(SETTLE_S * FS), int(cal_s * FS)
    offs_buf, cal_buf, rows = [], [], []
    offset, det, last_v = None, None, 0.0
    proc_time, n_proc, lead_off = 0.0, 0, False
    print("Reading DC offset (keep still)...")

    if plot:
        plt.ion()
        fig, axp = plt.subplots(figsize=(10, 3))
        hist = deque([0.0] * 500, maxlen=500)          # 5 s at 100 Hz
        (line_env,) = axp.plot(hist); line_thr = axp.axhline(0, color="r", ls="--")
        axp.set_title("Live RMS envelope (V)")

    try:
        for n, d in enumerate(sample_source(src, fast)):
            if seconds and n >= seconds * FS:
                break
            lead_off = np.isnan(d)
            volt = last_v if lead_off else d * VREF / 1023.0
            last_v = volt

            if offset is None:                          # phase 0: estimate DC offset
                offs_buf.append(volt)
                if len(offs_buf) == OFFSET_N:
                    offset = float(np.mean(offs_buf))
                    print(f"Settling filters ({SETTLE_S:.0f} s)...")
                continue

            t_a = time.perf_counter()
            y = volt - offset
            for s in chain:
                y = s.step(y)
            e = rms.step(y)
            proc_time += time.perf_counter() - t_a
            n_proc += 1
            m = n_proc

            if m <= settle_n:
                continue
            if m <= settle_n + cal_n:                   # phase 2: calibrate on rest
                cal_buf.append(e)
                if m == settle_n + 1:
                    print(f"CALIBRATING: keep muscle RELAXED for {cal_s:.0f} s...")
                if m == settle_n + cal_n:
                    thr_on, med = robust_threshold(np.array(cal_buf), k)
                    thr_off = med + 0.6 * (thr_on - med)
                    det = ActivityDetector(thr_on, thr_off)
                    print(f"Baseline {med:.4f} V, threshold {thr_on:.4f} V. "
                          f"Contract the muscle now. Ctrl-C to stop.")
                continue

            state = det.step(e)
            rows.append((n / FS, e, int(state)))
            if m % 100 == 0:                            # refresh display every 100 ms
                frac = min(e / (2 * det.thr_on), 1.0)
                bar = "#" * int(frac * 30) + "-" * (30 - int(frac * 30))
                tag = "ACTIVE" if state else "REST  "
                lo = "  LEAD-OFF!" if lead_off else ""
                print(f"\r[{tag}] env={e:.4f} V thr={det.thr_on:.4f} V |{bar}|{lo}   ",
                      end="", flush=True)
                if plot:
                    hist.append(e)
                    if m % 1000 == 0:
                        line_env.set_ydata(hist)
                        line_thr.set_ydata([det.thr_on, det.thr_on])
                        axp.set_ylim(0, max(max(hist), 2 * det.thr_on))
                        axp.set_facecolor("#d8f0d8" if state else "white")
                        plt.pause(0.001)
    except KeyboardInterrupt:
        pass

    print()
    if n_proc:
        us = proc_time / n_proc * 1e6
        print(f"Processed {n_proc} samples. Mean DSP time {us:.1f} us/sample "
              f"= {us / (1e6 / FS) * 100:.2f} % of the {1e3 / FS:.1f} ms sample period.")
    if det:
        print(f"Activations detected: {det.events}")
        print(f"Group delay of envelope window ~ {ENV_MS / 2:.0f} ms, "
              f"detector debounce {det.min_on * 1000 // FS} ms on / {det.min_off * 1000 // FS} ms off.")
    if rows:
        with open(log, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "envelope_V", "active"])
            w.writerows(rows)
        print(f"Log saved to {log}")


# =====================================================================
# 7. Recording and synthetic data
# =====================================================================
def record(port, seconds, out="emg_raw.csv"):
    import serial
    ser = serial.Serial(port, BAUD, timeout=1)
    time.sleep(2)                       # Uno resets when the port opens
    ser.reset_input_buffer()
    data, bad = [], 0
    print(f"Recording {seconds} s at {BAUD} baud.")
    print("Suggested: 5 s relaxed, 5 s hard contraction, 5 s relaxed.")
    t0 = time.time()
    last = t0
    try:
        while time.time() - t0 < seconds:
            line = ser.readline().decode(errors="ignore").strip()
            if line == "L":
                data.append(np.nan)
            elif line.isdigit():
                data.append(int(line))
            elif line:
                bad += 1
            if time.time() - last >= 1:
                print(f"  {time.time()-t0:4.1f} s  samples={len(data)}  bad lines={bad}")
                last = time.time()
            if time.time() - t0 > 5 and len(data) == 0:
                print("No valid data. Check BAUD and port.")
                break
    finally:
        ser.close()
    if not data:
        return
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["adc"])
        w.writerows([[d] for d in data])
    print(f"Saved {len(data)} samples to {out} (expected about {int(seconds*FS)})")


def synth(out="emg_synth.csv", seconds=15, seed=0):
    """5 s rest, 5 s contraction, 5 s rest, with drift, 50 Hz hum and ADC quantisation."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    t = np.arange(n) / FS
    gate = ((t >= 5) & (t < 10)).astype(float)
    v = (1.5 + 0.05 * np.sin(2 * np.pi * 0.2 * t) + 0.01 * np.sin(2 * np.pi * MAINS * t)
         + 0.004 * rng.standard_normal(n) + 0.15 * gate * rng.standard_normal(n))
    adc = np.clip(np.round(v / VREF * 1023), 0, 1023).astype(int)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["adc"])
        w.writerows([[a] for a in adc])
    print(f"Wrote {out} ({n} samples)")


# =====================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="EMG DSP project")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("record"); p.add_argument("port"); p.add_argument("seconds", type=float)
    p = sub.add_parser("analyze"); p.add_argument("csv"); p.add_argument("--no-plot", action="store_true")
    p = sub.add_parser("live"); p.add_argument("source", help="serial port or .csv to replay")
    p.add_argument("--seconds", type=float); p.add_argument("--fast", action="store_true",
                                                             help="replay CSV faster than real time")
    p.add_argument("--plot", action="store_true")
    p = sub.add_parser("synth"); p.add_argument("--out", default="emg_synth.csv")
    a = ap.parse_args()

    if a.cmd == "record":
        record(a.port, a.seconds)
    elif a.cmd == "analyze":
        analyze(a.csv, plot=not a.no_plot)
    elif a.cmd == "live":
        live(a.source, a.seconds, a.fast, a.plot)
    elif a.cmd == "synth":
        synth(a.out)
