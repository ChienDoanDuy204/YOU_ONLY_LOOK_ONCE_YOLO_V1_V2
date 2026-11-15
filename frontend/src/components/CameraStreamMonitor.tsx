"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Camera, FlipHorizontal, Square, Video } from "lucide-react";
import { drawDetectionsOverlay, getClassColor } from "@/lib/drawBoxes";
import { useStreamDetection } from "@/lib/useStreamDetection";

interface CameraStreamMonitorProps {
  confThreshold: number;
  iouThreshold: number;
  fps: number;
  onConfChange: (value: number) => void;
  onIouChange: (value: number) => void;
  onFpsChange: (value: number) => void;
}

export default function CameraStreamMonitor({
  confThreshold,
  iouThreshold,
  fps,
  onConfChange,
  onIouChange,
  onFpsChange,
}: CameraStreamMonitorProps) {
  const [cameraRunning, setCameraRunning] = useState(false);
  const [mirrorX, setMirrorX] = useState(true);
  const [cameraError, setCameraError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const { wsState, detections, sourceSize, latencyMs, effectiveFps, error, clearDetections } =
    useStreamDetection({
      videoRef,
      active: cameraRunning,
      fps,
      confThreshold,
      iouThreshold,
    });

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      for (const track of streamRef.current.getTracks()) {
        track.stop();
      }
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraRunning(false);
    clearDetections();
  }, [clearDetections]);

  const startCamera = useCallback(async () => {
    setCameraError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError("Browser camera API (`getUserMedia`) is unavailable on this origin.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraRunning(true);
    } catch (err) {
      setCameraError(
        err instanceof Error ? err.message : "Permission denied or no webcam found."
      );
      stopCamera();
    }
  }, [stopCamera]);

  const redrawOverlay = useCallback(() => {
    const canvas = overlayCanvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const srcW = sourceSize.width || video.videoWidth;
    const srcH = sourceSize.height || video.videoHeight;
    if (!srcW || !srcH) return;

    drawDetectionsOverlay(canvas, {
      sourceWidth: srcW,
      sourceHeight: srcH,
      detections: cameraRunning ? detections : [],
      mirrorX,
    });
  }, [cameraRunning, detections, mirrorX, sourceSize.height, sourceSize.width]);

  useEffect(() => {
    redrawOverlay();
  }, [redrawOverlay]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const observer = new ResizeObserver(() => redrawOverlay());
    observer.observe(video);
    return () => observer.disconnect();
  }, [cameraRunning, redrawOverlay]);

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  const combinedError = cameraError || error;

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
      {/* Camera Feed + Canvas Overlay */}
      <div className="space-y-4 lg:col-span-8">
        <div className="relative flex min-h-[440px] items-center justify-center overflow-hidden rounded-2xl border border-slate-800 bg-[#0F1623] p-4">
          <div
            className={`relative inline-block max-h-[620px] max-w-full ${
              cameraRunning ? "block" : "hidden"
            }`}
          >
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              className={`block max-h-[600px] max-w-full rounded-lg bg-black ${
                mirrorX ? "scale-x-[-1]" : ""
              }`}
            />
            <canvas
              ref={overlayCanvasRef}
              className="pointer-events-none absolute inset-0 h-full w-full"
            />
          </div>

          {!cameraRunning && (
            <div className="flex max-w-md flex-col items-center text-center">
              <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl border border-slate-700 bg-slate-900 text-amber-400">
                <Camera className="h-7 w-7" />
              </div>
              <h3 className="text-lg font-semibold text-white">Camera Stream Monitor</h3>
              <p className="mt-1.5 text-sm text-slate-400">
                Capture your webcam feed locally and stream frames to YOLOv2 over WebSocket with
                adjustable target FPS.
              </p>
              <button
                type="button"
                onClick={startCamera}
                className="mt-6 inline-flex items-center gap-2 rounded-xl bg-amber-400 px-5 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-amber-300"
              >
                <Video className="h-4 w-4" />
                Start Camera
              </button>
            </div>
          )}
        </div>

        {cameraRunning && (
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-800 bg-[#0F1623] px-4 py-3">
            <div className="flex items-center gap-2.5 font-mono text-xs text-slate-300">
              <span
                className={`h-2.5 w-2.5 rounded-full ${
                  wsState === "connected"
                    ? "animate-pulse bg-emerald-400"
                    : wsState === "connecting"
                    ? "bg-amber-400"
                    : "bg-rose-400"
                }`}
              />
              <span>CAMERA LIVE · WS: {wsState.toUpperCase()}</span>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setMirrorX((prev) => !prev)}
                className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium transition ${
                  mirrorX
                    ? "border-cyan-500/40 bg-cyan-500/10 text-cyan-300"
                    : "border-slate-700 bg-slate-900 text-slate-300"
                }`}
              >
                <FlipHorizontal className="h-3.5 w-3.5" />
                Mirror {mirrorX ? "On" : "Off"}
              </button>
              <button
                type="button"
                onClick={stopCamera}
                className="inline-flex items-center gap-1.5 rounded-lg border border-rose-500/40 bg-rose-500/10 px-3 py-1.5 text-xs font-medium text-rose-200 transition hover:bg-rose-500/20"
              >
                <Square className="h-3.5 w-3.5" />
                Stop Camera
              </button>
            </div>
          </div>
        )}

        {combinedError && (
          <div className="rounded-xl border border-rose-500/40 bg-rose-950/30 px-4 py-3 text-sm text-rose-200">
            {combinedError}
          </div>
        )}
      </div>

      {/* Right: FPS & NMS controls */}
      <div className="space-y-4 lg:col-span-4">
        <div className="rounded-2xl border border-slate-800 bg-[#0F1623] p-5">
          <h4 className="font-mono text-xs font-semibold uppercase tracking-wider text-slate-400">
            Camera FPS & NMS Controls
          </h4>

          <div className="mt-5 space-y-5">
            <div>
              <div className="mb-1.5 flex items-center justify-between text-sm">
                <span className="text-slate-200">Model Process FPS</span>
                <span className="font-mono font-semibold text-amber-300">{fps} FPS</span>
              </div>
              <input
                type="range"
                min={1}
                max={30}
                step={1}
                value={fps}
                onChange={(e) => onFpsChange(Number(e.target.value))}
                className="w-full accent-amber-400"
              />
              <p className="mt-1 font-mono text-[11px] text-slate-500">
                Capture throttle: {Math.round(1000 / Math.max(1, fps))} ms/frame
              </p>
            </div>

            <div>
              <div className="mb-1.5 flex items-center justify-between text-sm">
                <span className="text-slate-200">Confidence Threshold</span>
                <span className="font-mono font-semibold text-cyan-300">
                  {confThreshold.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min={0.05}
                max={0.95}
                step={0.01}
                value={confThreshold}
                onChange={(e) => onConfChange(Number(e.target.value))}
                className="w-full accent-cyan-400"
              />
            </div>

            <div>
              <div className="mb-1.5 flex items-center justify-between text-sm">
                <span className="text-slate-200">IoU NMS Threshold</span>
                <span className="font-mono font-semibold text-emerald-300">
                  {iouThreshold.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min={0.1}
                max={0.9}
                step={0.05}
                value={iouThreshold}
                onChange={(e) => onIouChange(Number(e.target.value))}
                className="w-full accent-emerald-400"
              />
            </div>
          </div>

          <div className="mt-6 grid grid-cols-3 gap-2.5 border-t border-slate-800 pt-4">
            <div className="rounded-xl bg-slate-950/70 p-2.5">
              <p className="font-mono text-[10px] text-slate-400">LATENCY</p>
              <p className="mt-1 font-mono text-sm font-semibold text-white">
                {latencyMs > 0 ? `${latencyMs.toFixed(0)} ms` : "--"}
              </p>
            </div>
            <div className="rounded-xl bg-slate-950/70 p-2.5">
              <p className="font-mono text-[10px] text-slate-400">ACTUAL FPS</p>
              <p className="mt-1 font-mono text-sm font-semibold text-emerald-300">
                {effectiveFps > 0 ? effectiveFps.toFixed(1) : "0.0"}
              </p>
            </div>
            <div className="rounded-xl bg-slate-950/70 p-2.5">
              <p className="font-mono text-[10px] text-slate-400">OBJECTS</p>
              <p className="mt-1 font-mono text-sm font-semibold text-cyan-300">
                {detections.length}
              </p>
            </div>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-[#0F1623] p-5">
          <h4 className="font-mono text-xs font-semibold uppercase tracking-wider text-slate-400">
            Live Camera Detections
          </h4>
          <div className="mt-3 max-h-[300px] space-y-2 overflow-y-auto pr-1">
            {detections.length > 0 ? (
              detections.map((det, idx) => (
                <div
                  key={`${det.class_id}-${idx}`}
                  className="flex items-center justify-between rounded-xl border border-slate-800/90 bg-slate-950/70 px-3 py-2 text-xs"
                >
                  <div className="flex items-center gap-2">
                    <span
                      className="h-2.5 w-2.5 rounded-full"
                      style={{ backgroundColor: getClassColor(det.class_id) }}
                    />
                    <span className="font-semibold capitalize text-white">{det.class_name}</span>
                  </div>
                  <span className="font-mono font-semibold text-cyan-300">
                    {(det.score * 100).toFixed(1)}%
                  </span>
                </div>
              ))
            ) : (
              <p className="py-6 text-center text-xs text-slate-500">
                {cameraRunning
                  ? "No detections above confidence threshold."
                  : "Click Start Camera to inspect live webcam detections."}
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

