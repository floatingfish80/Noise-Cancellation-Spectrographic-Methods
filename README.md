# Noise Cancellation Using Spectrographic Analysis

## Description
Developed a comprehensive audio signal processing pipeline in Python that captures, analyzes, filters, and enhances recorded audio using real-time techniques.

- Recorded mono audio at 44.1 kHz using the `sounddevice` library.
- Performed time-domain, frequency-domain, and **spectrographic** analysis for full signal insight.
- Applied a **bandpass Butterworth filter** (300 Hz–4000 Hz) to suppress low and high-frequency noise while preserving speech components.
- Built an **amplification module** with normalization to avoid clipping.
- Detected **dominant frequency components** and performed **targeted noise band analysis** (e.g., 50–300 Hz) for adaptive profiling.
- Exported audio to `.wav` and enabled in-notebook playback with `IPython.display.Audio`.

## Why Spectrographic Analysis?
Spectrograms show how frequency content varies over time — a critical feature for detecting and understanding non-stationary noise patterns that are not visible in static plots. This allows:

- Isolation of specific noise bands (e.g., hum, hiss, or chatter).
- Visual validation of filter effectiveness.
- Design of time-sensitive or adaptive filters.

Spectrographic insight is essential for tuning filters and verifying the success of noise cancellation techniques in dynamic environments.

## Libraries Used
- **NumPy** – Numerical computations  
- **SciPy** – Signal processing (`fft`, `butter`, `filtfilt`) and I/O (`wavfile.write`)  
- **Matplotlib** – Plotting waveform, frequency spectrum, and spectrograms  
- **SoundDevice** – Real-time audio recording  
- **IPython.display** – Audio playback inside Jupyter notebooks
