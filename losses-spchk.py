#!/usr/bin/env python3
"""
audio_quality_check.py
=======================
Analyzes a music file: shows full metadata, generates a spectrogram, and
tries to detect signs of transcoding (e.g. MP3 -> FLAC, i.e. "fake lossless").

Install dependencies:
    pip install librosa mutagen matplotlib numpy scipy --break-system-packages

Usage:
    python audio_quality_check.py path/to/file.flac
    python audio_quality_check.py path/to/file.mp3 --output spectrogram.png
"""

import argparse
import os
import sys

import numpy as np


def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def show_metadata(path):
    """Displays all available file metadata (tags, format, bitrate, etc.)."""
    from mutagen import File as MutagenFile

    print_section("FILE METADATA")

    f = MutagenFile(path)
    if f is None:
        print("Could not read metadata (unsupported format?).")
        return None

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"File:           {path}")
    print(f"Size:           {size_mb:.2f} MB")

    info = f.info
    duration = getattr(info, "length", None)
    bitrate = getattr(info, "bitrate", None)
    sample_rate = getattr(info, "sample_rate", None)
    channels = getattr(info, "channels", None)
    bits_per_sample = getattr(info, "bits_per_sample", None)

    if duration:
        m, s = divmod(int(duration), 60)
        print(f"Duration:       {m}:{s:02d}")
    if bitrate:
        print(f"Bitrate:        {bitrate // 1000} kb/s")
    if sample_rate:
        print(f"Sample rate:    {sample_rate} Hz")
    if channels:
        print(f"Channels:       {channels}")
    if bits_per_sample:
        print(f"Bit depth:      {bits_per_sample} bit")

    # VBR/CBR mode if mutagen knows it (MP3)
    if hasattr(info, "bitrate_mode"):
        print(f"Bitrate mode:   {info.bitrate_mode}")

    print("\n-- Tags --")
    if f.tags:
        for key, value in f.tags.items():
            print(f"{key}: {value}")
    else:
        print("No tags found.")

    return {
        "duration": duration,
        "bitrate": bitrate,
        "sample_rate": sample_rate,
        "channels": channels,
        "bits_per_sample": bits_per_sample,
    }


def detect_cutoff_robust(S_db, freqs, sr):
    """
    Robust detection of the frequency-band cutoff in a spectrogram.

    Unlike a naive "single threshold on the averaged spectrum" approach,
    this function:
    1. Analyzes EACH time frame separately (not the track-wide average),
       discarding silent/inactive frames that would skew the result.
    2. Computes the MEDIAN and SPREAD (MAD) of the cutoff point over time:
       - an artificial low-pass filter from a lossy codec has a PRACTICALLY
         CONSTANT cutoff point throughout the whole track (it's just a
         digital filter),
       - natural musical content has a cutoff point that FLUCTUATES
         depending on what's currently playing (quiet vocals vs loud
         cymbals).
    3. Measures the STEEPNESS of the edge (how many dB the signal drops
       within a 1.5 kHz band from the detected cutoff point) — lossy codecs
       create an almost vertical "wall", while natural high-frequency
       roll-off is gentle and spread over a wider range.
    """
    n_freqs, n_frames = S_db.shape
    nyquist = sr / 2

    # Global noise floor (5th percentile of the whole spectrogram)
    noise_floor = float(np.percentile(S_db, 5))

    # Discard quiet frames (intro, fade-outs, transitions) — only analyze
    # segments within 40 dB of the track's loudest moment.
    frame_peak = S_db.max(axis=0)
    track_peak = float(frame_peak.max())
    active_mask = frame_peak > (track_peak - 40)
    if active_mask.sum() < 10:
        active_mask = np.ones(n_frames, dtype=bool)  # safeguard for very short/quiet files

    # Smooth the spectrum along the frequency axis (~300 Hz window) so that
    # individual tonal peaks/dips don't corrupt edge detection.
    freq_resolution = freqs[1] - freqs[0]
    smooth_bins = max(1, int(round(300 / freq_resolution)))
    kernel = np.ones(smooth_bins) / smooth_bins
    S_db_smooth = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="same"), 0, S_db)

    delta_db = 10
    threshold = noise_floor + delta_db
    offset_hz = 1500  # window used to measure edge steepness

    cutoffs = []
    drop_values = []

    for i in np.where(active_mask)[0]:
        col = S_db_smooth[:, i]
        above = np.where(col > threshold)[0]
        if len(above) == 0:
            continue
        cutoff_idx = above[-1]
        cutoff_freq = freqs[cutoff_idx]
        cutoffs.append(cutoff_freq)

        target_freq = min(cutoff_freq + offset_hz, nyquist - 1)
        target_idx = min(np.searchsorted(freqs, target_freq), n_freqs - 1)
        drop_values.append(col[cutoff_idx] - col[target_idx])

    if len(cutoffs) == 0:
        return {
            "median_cutoff": nyquist, "mad": 0.0, "steepness_per_khz": 0.0,
            "n_active_frames": 0, "noise_floor": noise_floor,
        }

    cutoffs = np.array(cutoffs)
    drop_values = np.array(drop_values)
    median_cutoff = float(np.median(cutoffs))
    mad = float(np.median(np.abs(cutoffs - median_cutoff)))
    steepness_per_khz = float(np.median(drop_values)) / (offset_hz / 1000)

    return {
        "median_cutoff": median_cutoff,
        "mad": mad,
        "steepness_per_khz": steepness_per_khz,
        "n_active_frames": int(len(cutoffs)),
        "noise_floor": noise_floor,
    }


def print_verdict(stats, nyquist):
    """Issues a verdict based on three independent signals: cutoff position,
    its stability over time (MAD), and edge steepness."""
    median_cutoff = stats["median_cutoff"]
    mad = stats["mad"]
    steepness = stats["steepness_per_khz"]
    n_frames = stats["n_active_frames"]

    print_section("DETAILED ANALYSIS")
    print(f"Active frames analyzed:    {n_frames}")
    print(f"Median cutoff point:       ~{median_cutoff:.0f} Hz")
    print(f"Cutoff spread over time:   ±{mad:.0f} Hz (MAD)")
    print(f"Edge steepness:            {steepness:.1f} dB/kHz")

    ratio = median_cutoff / nyquist

    known_cutoffs = {128: 16000, 160: 17500, 192: 19000, 256: 20000, 320: 20500}

    print_section("VERDICT — IS THIS TRANSCODED?")

    if ratio >= 0.97:
        print("OK  NO CLEAR SIGNS OF TRANSCODING.")
        print("    The frequency band is used almost up to the Nyquist limit —")
        print("    the file looks like genuine lossless / high-quality audio.")
        return

    # "Stability over time" threshold: a small MAD value means the cutoff
    # point barely changes throughout the whole track = sign of an
    # ARTIFICIAL digital filter.
    is_consistent = mad < 400
    # Steepness threshold: lossy codecs cut off almost vertically.
    is_steep = steepness > 8

    if is_consistent and is_steep:
        print("WARNING  HIGH PROBABILITY OF TRANSCODING / FAKE LOSSLESS.")
        print(f"    The cutoff point (~{median_cutoff:.0f} Hz) is nearly IDENTICAL")
        print(f"    throughout the whole track (spread of only ±{mad:.0f} Hz) and")
        print(f"    drops very steeply ({steepness:.1f} dB/kHz). This is a classic")
        print("    sign of a digital low-pass filter from a lossy codec (e.g. MP3),")
        print("    later saved into a lossless container.")
        closest_bitrate = min(known_cutoffs, key=lambda b: abs(known_cutoffs[b] - median_cutoff))
        print(f"    Closest match: a source MP3 at ~{closest_bitrate} kb/s.")
    elif is_consistent or is_steep:
        print("MAYBE  INCONCLUSIVE — partial signs, but not definitive.")
        if is_consistent and not is_steep:
            print(f"    The cutoff point is stable over time (±{mad:.0f} Hz), but the")
            print("    edge drops gently — this could be a mastering/recording")
            print("    filter rather than necessarily lossy compression.")
        else:
            print(f"    The edge is fairly steep ({steepness:.1f} dB/kHz), but the")
            print(f"    cutoff point shifts over time (±{mad:.0f} Hz) — this fits")
            print("    natural musical content better than a fixed filter.")
        print("    Recommended: visually compare the spectrogram or listen to the file.")
    else:
        print("OK  LIKELY NO TRANSCODING.")
        print(f"    Although the band ends below Nyquist (~{median_cutoff:.0f} Hz),")
        print(f"    this point CLEARLY SHIFTS over time (±{mad:.0f} Hz) and the edge")
        print(f"    is gentle ({steepness:.1f} dB/kHz) — this is typical of natural")
        print("    musical content (e.g. ballads, acoustic recordings), not a")
        print("    rigid digital filter from a lossy codec.")

    print("\nNote: this is a heuristic analysis (similar to tools like 'Spek' or the")
    print("spectro plugin in foobar2000). The most reliable method is still comparing")
    print("against a known, original source, if one is available.")


def analyze_spectrum(path):
    """Computes the spectrum (STFT) and analyzes it for band cutoff (transcoding).
    Also returns raw data, in case the spectrogram needs to be drawn later."""
    import librosa

    print_section("SPECTRUM ANALYSIS")

    # sr=None -> keep the file's original sample rate
    y, sr = librosa.load(path, sr=None, mono=True)

    # Larger n_fft = better frequency resolution, needed for precise
    # detection of the band cutoff edge.
    n_fft = 8192
    hop_length = 2048
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    nyquist = sr / 2

    print(f"File sample rate:          {sr} Hz (Nyquist = {nyquist:.0f} Hz)")

    stats = detect_cutoff_robust(S_db, freqs, sr)
    print_verdict(stats, nyquist)

    return {"y": y, "sr": sr, "S_db": S_db, "hop_length": hop_length}


def save_spectrogram(path, spectrum_data, output_path):
    """Draws and saves the spectrogram from data computed earlier."""
    import librosa.display
    import matplotlib.pyplot as plt

    sr = spectrum_data["sr"]
    S_db = spectrum_data["S_db"]
    hop_length = spectrum_data["hop_length"]

    fig, ax = plt.subplots(figsize=(14, 6))
    img = librosa.display.specshow(
        S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz", ax=ax, cmap="magma"
    )
    ax.set_title(f"Spectrogram: {os.path.basename(path)}")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nSpectrogram saved to: {os.path.abspath(output_path)}")


def main():
    parser = argparse.ArgumentParser(description="Analyze audio file quality and detect transcoding.")
    parser.add_argument("file", help="Path to the audio file (mp3, flac, wav, m4a, ogg, ...)")
    parser.add_argument(
        "--output", "-o", default=None,
        help="Spectrogram PNG file name (default: <filename>_spectrogram.png in the current directory)"
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Don't ask, generate the spectrogram file immediately"
    )
    parser.add_argument(
        "--no-spectrogram", action="store_true",
        help="Don't ask, never generate the spectrogram file"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print(f"Error: file '{args.file}' does not exist.")
        sys.exit(1)

    show_metadata(args.file)
    spectrum_data = analyze_spectrum(args.file)

    if args.no_spectrogram:
        return

    # Default output file name, saved in the directory the script was run
    # from (i.e. the current working directory, os.getcwd()).
    default_name = os.path.splitext(os.path.basename(args.file))[0] + "_spectrogram.png"
    output_path = args.output or os.path.join(os.getcwd(), default_name)

    if args.yes:
        save_spectrogram(args.file, spectrum_data, output_path)
        return

    print_section("GENERATE SPECTROGRAM FILE")
    answer = input(f"Do you want to save the spectrogram as a PNG image? ({output_path}) [y/N]: ").strip().lower()
    if answer in ("y", "yes"):
        save_spectrogram(args.file, spectrum_data, output_path)
    else:
        print("Skipped saving the spectrogram.")


if __name__ == "__main__":
    main()
