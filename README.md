[README.md](https://github.com/user-attachments/files/29196945/README.md)
# Audio Quality Check

A command-line tool that inspects a music file's metadata, renders a spectrogram, and uses a multi-signal heuristic to detect **transcoding** (e.g. an MP3 re-encoded into FLAC, often called "fake lossless").

## Features

- 📋 **Full metadata dump** — format, bitrate, sample rate, bit depth, channels, duration, VBR/CBR mode, and all embedded tags (artist, album, etc.)
- 📊 **Spectrogram generation** — high-resolution STFT spectrogram saved as a PNG, similar to tools like [Spek](http://spek.cc/) or foobar2000's spectro plugin
- 🔍 **Transcoding detection** — analyzes the frequency content frame-by-frame (not just an averaged spectrum) and combines three independent signals:
  - **Median cutoff frequency** — where the usable frequency band actually ends
  - **Stability over time (MAD)** — a real lossy-codec low-pass filter sits at almost the exact same frequency throughout the whole track; natural musical content fluctuates depending on what's playing
  - **Edge steepness (dB/kHz)** — lossy codecs cut off in a near-vertical "wall"; natural high-frequency roll-off is gradual

This combination avoids the most common false positive of naive spectrogram analysis: flagging quiet, vocal-heavy, or ballad-style tracks as "transcoded" just because they naturally have little energy above ~17–18 kHz.

## Installation

```bash
git clone https://github.com/yourusername/audio-quality-check.git
cd audio-quality-check
pip install librosa mutagen matplotlib numpy scipy --break-system-packages
```

> `--break-system-packages` is only needed on systems with an externally-managed Python (e.g. recent Debian/Ubuntu). Using a virtual environment is recommended:
> ```bash
> python -m venv venv
> source venv/bin/activate   # Windows: venv\Scripts\activate
> pip install librosa mutagen matplotlib numpy scipy
> ```

## Usage

Basic run — prints metadata and the transcoding verdict, then asks whether to save a spectrogram image:

```bash
python audio_quality_check.py "path/to/song.flac"
```

```
Do you want to save the spectrogram as a PNG image? (./song_spectrogram.png) [y/N]:
```

### Options

| Flag | Description |
|---|---|
| `-o`, `--output <path>` | Custom output filename/path for the spectrogram PNG |
| `-y`, `--yes` | Skip the prompt, always generate the spectrogram |
| `--no-spectrogram` | Skip the prompt, never generate the spectrogram |

```bash
# Save with a custom name, no prompt
python audio_quality_check.py "song.mp3" --output spec.png --yes

# Metadata + verdict only, no image at all
python audio_quality_check.py "song.wav" --no-spectrogram
```

The spectrogram (if generated) is saved in the **current working directory** by default — i.e. wherever you run the script from, not necessarily next to the audio file.

## Example output

```
============================================================
FILE METADATA
============================================================
File:           Cruel Angel's Thesis.flac
Size:           34.18 MB
Duration:       4:01
Sample rate:    44100 Hz
Channels:       2
Bit depth:      16 bit

============================================================
SPECTRUM ANALYSIS
============================================================
File sample rate:          44100 Hz (Nyquist = 22050 Hz)

============================================================
DETAILED ANALYSIS
============================================================
Active frames analyzed:    1842
Median cutoff point:       ~17937 Hz
Cutoff spread over time:   ±1850 Hz (MAD)
Edge steepness:            6.3 dB/kHz

============================================================
VERDICT — IS THIS TRANSCODED?
============================================================
OK  LIKELY NO TRANSCODING.
    Although the band ends below Nyquist (~17937 Hz),
    this point CLEARLY SHIFTS over time (±1850 Hz) and the edge
    is gentle (6.3 dB/kHz) — this is typical of natural
    musical content (e.g. ballads, acoustic recordings), not a
    rigid digital filter from a lossy codec.
```

## How the detection works

1. The track is loaded at its original sample rate and converted to an STFT spectrogram (`n_fft=8192`, `hop_length=2048` for good frequency resolution).
2. Silent/quiet frames (more than 40 dB below the track's peak) are discarded so intros, fades, and pauses don't skew the result.
3. For each remaining frame, the spectrum is smoothed (~300 Hz window) and the highest frequency still above the noise floor is recorded as that frame's "cutoff point."
4. The **median** and **MAD (median absolute deviation)** of all per-frame cutoffs are computed — MAD is the stability signal.
5. **Edge steepness** is measured as the dB drop across a 1.5 kHz band starting at the cutoff.
6. A verdict is only issued as "likely transcoded" when **both** the cutoff is stable over time (MAD < 400 Hz) **and** the edge is steep (> 8 dB/kHz). Either signal alone is treated as inconclusive.

## Limitations

- This is a **heuristic**, not a forensic tool. It cannot replace a direct A/B comparison with a known original source.
- Some genuinely lossless recordings (especially from older masters, vinyl rips, or sources that were already low-pass filtered during mastering) can still show a stable, steep cutoff and trigger a false positive.
- Some skillfully transcoded files (e.g. upsampled with a gentle filter) may produce a false negative.
- Works on any format supported by `librosa`/`mutagen` (MP3, FLAC, WAV, M4A/AAC, OGG, etc.).

## Requirements

- Python 3.8+
- [librosa](https://librosa.org/)
- [mutagen](https://mutagen.readthedocs.io/)
- matplotlib
- numpy
- scipy

## License

MIT
