const VIDEO_TYPES = ["video/webm;codecs=vp8", "video/webm;codecs=vp9", "video/webm", "video/mp4"];
const AUDIO_TYPES = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];
const CHUNK_MS = 4000;
const VIDEO_BITRATE = 2_500_000;
const MAX_ATTEMPTS = 6;

export function pickType(kind) {
  if (typeof MediaRecorder === "undefined") return null;
  const types = kind === "video" ? VIDEO_TYPES : AUDIO_TYPES;
  return types.find((type) => MediaRecorder.isTypeSupported(type)) ?? "";
}

export function mediaSupport() {
  return {
    camera: Boolean(navigator.mediaDevices?.getUserMedia),
    recorder: typeof MediaRecorder !== "undefined",
    secure: window.isSecureContext,
  };
}

export async function openCamera(deviceId = null) {
  const video = {
    width: { ideal: 1280 },
    height: { ideal: 720 },
    frameRate: { ideal: 30 },
  };
  if (deviceId) video.deviceId = { exact: deviceId };
  else video.facingMode = "user";
  return navigator.mediaDevices.getUserMedia({ video, audio: false });
}

export async function openMicrophone() {
  return navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: true },
    video: false,
  });
}

export async function listCameras() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices.filter((device) => device.kind === "videoinput");
}

export function stopStream(stream) {
  stream?.getTracks().forEach((track) => track.stop());
}

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

class ChunkUploader {
  constructor(kind, mime) {
    this.kind = kind;
    this.mime = mime;
    this.seq = 0;
    this.chain = Promise.resolve();
    this.bytes = 0;
    this.uploaded = 0;
    this.failed = null;
    this.pending = 0;
  }

  push(blob) {
    if (!blob.size) return;
    const seq = this.seq++;
    this.bytes += blob.size;
    this.pending++;
    this.chain = this.chain.then(() => this.send(seq, blob)).finally(() => this.pending--);
  }

  async send(seq, blob) {
    if (this.failed) return;
    const url = `/api/session/media/${this.kind}?seq=${seq}&mime=${encodeURIComponent(this.mime)}`;
    for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
      try {
        const response = await fetch(url, { method: "POST", body: blob });
        if (response.ok) {
          this.uploaded += blob.size;
          return;
        }
        if (response.status === 409) {
          this.failed = (await response.json().catch(() => ({}))).error || "Upload rejected";
          return;
        }
      } catch {
      }
      await wait(500 * attempt);
    }
    this.failed = `Could not upload ${this.kind} chunk ${seq}`;
  }

  drain() {
    return this.chain;
  }
}

export class Recorder {
  constructor(stream, kind) {
    this.kind = kind;
    this.mime = pickType(kind);
    const options = { mimeType: this.mime || undefined };
    if (kind === "video") options.videoBitsPerSecond = VIDEO_BITRATE;
    this.recorder = new MediaRecorder(stream, options);
    this.mime = this.recorder.mimeType || this.mime;
    this.uploader = new ChunkUploader(kind, this.mime);
    this.recorder.addEventListener("dataavailable", (event) => this.uploader.push(event.data));
    this.startedAt = null;
    this.stoppedAt = null;
    this.error = null;
    this.recorder.addEventListener("error", (event) => {
      this.error = event.error?.message || "Recorder error";
    });
  }

  start() {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`The ${this.kind} recorder did not start`)), 5000);
      this.recorder.addEventListener(
        "start",
        (event) => {
          clearTimeout(timer);
          this.startedAt = event.timeStamp;
          resolve(this.startedAt);
        },
        { once: true }
      );
      this.recorder.start(CHUNK_MS);
    });
  }

  async stop() {
    if (this.recorder.state !== "inactive") {
      const stopped = new Promise((resolve) => this.recorder.addEventListener("stop", resolve, { once: true }));
      this.recorder.stop();
      await stopped;
    }
    this.stoppedAt ??= performance.now();
    await this.uploader.drain();
    return this.info();
  }

  progress() {
    const { bytes, uploaded } = this.uploader;
    return bytes ? uploaded / bytes : 1;
  }

  info() {
    return {
      mime: this.mime,
      start_s: this.startedAt === null ? null : this.startedAt / 1000,
      start_epoch_ms: this.startedAt === null ? null : performance.timeOrigin + this.startedAt,
      duration_s: this.startedAt === null ? null : (this.stoppedAt - this.startedAt) / 1000,
      chunks: this.uploader.seq,
      bytes: this.uploader.bytes,
      upload_error: this.uploader.failed,
      recorder_error: this.error,
    };
  }
}

export function levelMeter(stream) {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass || !stream?.getAudioTracks().length) return null;
  const context = new AudioContextClass();
  const analyser = context.createAnalyser();
  analyser.fftSize = 512;
  context.createMediaStreamSource(stream).connect(analyser);
  const samples = new Uint8Array(analyser.fftSize);
  return {
    resume: () => context.resume(),
    level() {
      analyser.getByteTimeDomainData(samples);
      let sum = 0;
      for (const value of samples) sum += ((value - 128) / 128) ** 2;
      return Math.min(1, Math.sqrt(sum / samples.length) * 4);
    },
    close: () => context.close(),
  };
}

export function grabFrame(video, canvas, maxWidth = 640) {
  const scale = Math.min(1, maxWidth / video.videoWidth);
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  return context;
}

export function regionBrightness(context, box) {
  const { width, height } = context.canvas;
  const x = Math.max(0, Math.round(box.x * width));
  const y = Math.max(0, Math.round(box.y * height));
  const w = Math.max(2, Math.min(width - x, Math.round(box.w * width)));
  const h = Math.max(1, Math.min(height - y, Math.round(box.h * height)));
  const pixels = context.getImageData(x, y, w, h).data;
  let left = 0;
  let right = 0;
  let leftCount = 0;
  let rightCount = 0;
  for (let row = 0; row < h; row += 2) {
    for (let col = 0; col < w; col += 2) {
      const i = (row * w + col) * 4;
      const luma = 0.299 * pixels[i] + 0.587 * pixels[i + 1] + 0.114 * pixels[i + 2];
      if (col < w / 2) {
        left += luma;
        leftCount++;
      } else {
        right += luma;
        rightCount++;
      }
    }
  }
  left /= leftCount || 1;
  right /= rightCount || 1;
  return { mean: (left + right) / 2, left, right };
}
