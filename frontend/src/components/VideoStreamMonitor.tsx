"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Pause, Play, Upload, Video } from "lucide-react";
import { drawDetectionsOverlay, getClassColor } from "@/lib/drawBoxes";
import { useStreamDetection } from "@/lib/useStreamDetection";

interface VideoStreamMonitorProps {
  confThreshold: number;
  iouThreshold: number;
  fps: number;
  onConfChange: (value: number) => void;
  onIouChange: (value: number) => void;
  onFpsChange: (value: number) => void;
}

export default function VideoStreamMonitor({
  confThreshold,
  iouThreshold,
  fps,
  onConfChange,
  onIouChange,
  onFpsChange,
}: VideoStreamMonitorProps) {
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoName, setVideoName] = useState<string>("");
  const [aiEnabled, setAiEnabled] = useState<boolean>(true);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const { wsState, detections, sourceSize, latencyMs, effectiveFps, error, clearDetections } =
    useStreamDetection({
      videoRef,
      active: Boolean(videoUrl) && aiEnabled,
      fps,
      confThreshold,
      iouThreshold,
    });

  const handleVideoFile = useCallback(
    (file: File) => {
      clearDetections();
      setVideoName(file.name);
      setVideoUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return URL.createObjectURL(file);
      });
    },
    [clearDetections]
  );

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
      detections: aiEnabled ? detections : [],
    });
  }, [aiEnabled, detections, sourceSize.height, sourceSize.width]);

  useEffect(() => {
    redrawOverlay();
  }, [redrawOverlay]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const observer = new ResizeObserver(() => redrawOverlay());
    observer.observe(video);
    return () => observer.disconnect();
  }, [videoUrl, redrawOverlay]);

  useEffect(() => {
    return () => {
      if (videoUrl) URL.revokeObjectURL(videoUrl);
    };
  }, [videoUrl]);

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
      {/* Video Player + Canvas Overlay */}
      <div className="space-y-4 lg:col-span-8">
        <div className="relative flex min-h-[440px] items-center justify-center overflow-hidden rounded-2xl border border-slate-800 bg-[#0F1623] p-4">
          {videoUrl ? (
            <div className="relative inline-block max-h-[620px] max-w-full">
              <video
                ref={videoRef}
                src={videoUrl}
                controls
                playsInline
                loop
                className="block max-h-[600px] max-w-full rounded-lg bg-black"
              />
              <canvas
                ref={overlayCanvasRef}
                className="pointer-events-none absolute inset-0 h-full w-full"
              />
            </div>
          ) : (
            <div className="flex max-w-md flex-col items-center text-center">
              <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl border border-slate-700 bg-slate-900 text-emerald-400">
                <Video className="h-7 w-7" />
              </div>
              <h3 className="text-lg font-semibold text-white">Process Video Stream Monitor</h3>
              <p className="mt-1.5 text-sm text-slate-400">
                Select a local `.mp4`, `.webm`, or `.mov` video. Frames are sampled at your
                selected FPS and streamed over WebSocket to YOLOv2.
              </p>
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="mt-6 inline-flex items-center gap-2 rounded-xl bg-emerald-400 px-5 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-emerald-300"
              >
                <Upload className="h-4 w-4" />
                Select Video File
              </button>
            </div>
          )}

          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleVideoFile(file);
            }}
          />
        </div>

        {videoUrl && (
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-800 bg-[#0F1623] px-4 py-3">
            <div className="flex items-center gap-3 font-mono text-xs text-slate-300">
              <span
                className={`h-2.5 w-2.5 rounded-full ${
                  wsState === "connected"
                    ? "bg-emerald-400"
                    : wsState === "connecting"
                    ? "bg-amber-400"
                    : "bg-slate-500"
                }`}
              />
              <span className="truncate max-w-[260px]">{videoName}</span>
              <span className="text-slate-500">· WS: {wsState.toUpperCase()}</span>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setAiEnabled((prev) => !prev)}
                className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium transition ${
                  aiEnabled
                    ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
                    : "border-slate-700 bg-slate-900 text-slate-300"
                }`}
              >
                {aiEnabled ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
                {aiEnabled ? "Pause AI Overlay" : "Resume AI Overlay"}
              </button>
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs font-medium text-slate-200 transition hover:border-cyan-400"
              >
                <Upload className="h-3.5 w-3.5" />
                Change Video
              </button>
            </div>
          </div>
        )}

        {error && (
          <div className="rounded-xl border border-rose-500/40 bg-rose-950/30 px-4 py-3 text-sm text-rose-200">
            {error}
          </div>
        )}
      </div>

      {/* Right: FPS & NMS controls */}
      <div className="space-y-4 lg:col-span-4">
        <div className="rounded-2xl border border-slate-800 bg-[#0F1623] p-5">
          <h4 className="font-mono text-xs font-semibold uppercase tracking-wider text-slate-400">
            Stream & NMS Controls
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
                Dispatch interval: {Math.round(1000 / Math.max(1, fps))} ms/frame
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
            Live Stream Detections
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
                {videoUrl
                  ? "Play the video to stream frames to YOLOv2."
                  : "Load a video file to start real-time detection."}
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

