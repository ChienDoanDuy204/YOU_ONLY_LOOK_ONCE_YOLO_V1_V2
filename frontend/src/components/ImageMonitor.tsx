"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ImageIcon, Loader2, Sparkles, Upload } from "lucide-react";
import { predictImageFile } from "@/lib/api";
import { drawDetectionsOverlay, getClassColor } from "@/lib/drawBoxes";
import type { DetectionResponse } from "@/lib/types";

interface ImageMonitorProps {
  confThreshold: number;
  iouThreshold: number;
  onConfChange: (value: number) => void;
  onIouChange: (value: number) => void;
}

export default function ImageMonitor({
  confThreshold,
  iouThreshold,
  onConfChange,
  onIouChange,
}: ImageMonitorProps) {
  const [selectedFile, setSelectedFile] = useState<Blob | null>(null);
  const [fileName, setFileName] = useState<string>("");
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const imgRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const runDetection = useCallback(
    async (file: Blob, conf: number, iou: number) => {
      setLoading(true);
      setError(null);
      try {
        const res = await predictImageFile(file, {
          confThreshold: conf,
          iouThreshold: iou,
        });
        setResult(res);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to run detection");
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const handleSelectFile = useCallback(
    (file: File) => {
      setSelectedFile(file);
      setFileName(file.name);
      setResult(null);
      setPreviewUrl((prev) => {
        if (prev && prev.startsWith("blob:")) {
          URL.revokeObjectURL(prev);
        }
        return URL.createObjectURL(file);
      });
      void runDetection(file, confThreshold, iouThreshold);
    },
    [confThreshold, iouThreshold, runDetection]
  );

  const handleLoadSample = useCallback(async () => {
    try {
      setError(null);
      const response = await fetch("/demo-scene.jpg");
      const blob = await response.blob();
      setSelectedFile(blob);
      setFileName("demo-scene.jpg");
      setPreviewUrl((prev) => {
        if (prev && prev.startsWith("blob:")) {
          URL.revokeObjectURL(prev);
        }
        return "/demo-scene.jpg";
      });
      await runDetection(blob, confThreshold, iouThreshold);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load sample image");
    }
  }, [confThreshold, iouThreshold, runDetection]);

  // Debounced re-run when confidence/IoU sliders change
  useEffect(() => {
    if (!selectedFile) return;
    const timer = setTimeout(() => {
      void runDetection(selectedFile, confThreshold, iouThreshold);
    }, 180);
    return () => clearTimeout(timer);
  }, [selectedFile, confThreshold, iouThreshold, runDetection]);

  const redrawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;
    if (!result) {
      const ctx = canvas.getContext("2d");
      ctx?.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    drawDetectionsOverlay(canvas, {
      sourceWidth: result.width,
      sourceHeight: result.height,
      detections: result.detections,
    });
  }, [result]);

  useEffect(() => {
    redrawCanvas();
  }, [redrawCanvas]);

  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;
    const observer = new ResizeObserver(() => redrawCanvas());
    observer.observe(img);
    return () => observer.disconnect();
  }, [previewUrl, redrawCanvas]);

  useEffect(() => {
    return () => {
      if (previewUrl && previewUrl.startsWith("blob:")) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
      {/* Left: Image viewport */}
      <div className="space-y-4 lg:col-span-8">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setIsDragging(false);
            const dropped = e.dataTransfer.files?.[0];
            if (dropped && dropped.type.startsWith("image/")) {
              handleSelectFile(dropped);
            }
          }}
          className={`relative flex min-h-[440px] items-center justify-center overflow-hidden rounded-2xl border bg-[#0F1623] p-4 transition ${
            isDragging ? "border-cyan-400 bg-cyan-950/20" : "border-slate-800"
          }`}
        >
          {previewUrl ? (
            <div className="relative inline-block max-h-[640px] max-w-full">
              <img
                ref={imgRef}
                src={previewUrl}
                alt={fileName || "Uploaded image"}
                onLoad={redrawCanvas}
                className="block max-h-[620px] max-w-full rounded-lg object-contain"
              />
              <canvas
                ref={canvasRef}
                className="pointer-events-none absolute inset-0 h-full w-full"
              />
              {loading && (
                <div className="absolute right-3 top-3 inline-flex items-center gap-2 rounded-full border border-cyan-500/40 bg-slate-950/85 px-3 py-1.5 font-mono text-xs text-cyan-300">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  <span>Running YOLOv2...</span>
                </div>
              )}
            </div>
          ) : (
            <div className="flex max-w-md flex-col items-center text-center">
              <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl border border-slate-700 bg-slate-900 text-cyan-400">
                <ImageIcon className="h-7 w-7" />
              </div>
              <h3 className="text-lg font-semibold text-white">Process Image Monitor</h3>
              <p className="mt-1.5 text-sm text-slate-400">
                Drop an image here, pick a file from disk, or test immediately with the PASCAL VOC
                sample scene.
              </p>
              <div className="mt-6 flex flex-wrap justify-center gap-3">
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="inline-flex items-center gap-2 rounded-xl bg-cyan-400 px-4 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-cyan-300"
                >
                  <Upload className="h-4 w-4" />
                  Upload Image
                </button>
                <button
                  type="button"
                  onClick={handleLoadSample}
                  className="inline-flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-900 px-4 py-2.5 text-sm font-medium text-slate-200 transition hover:border-emerald-400/60 hover:text-emerald-300"
                >
                  <Sparkles className="h-4 w-4" />
                  Test Sample Scene
                </button>
              </div>
            </div>
          )}

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleSelectFile(file);
            }}
          />
        </div>

        {previewUrl && (
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-800 bg-[#0F1623] px-4 py-3">
            <div className="font-mono text-xs text-slate-400">
              FILE: <span className="text-slate-200">{fileName}</span>
              {result && (
                <span className="ml-3 text-slate-500">
                  ({result.width}×{result.height})
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs font-medium text-slate-200 transition hover:border-cyan-400"
              >
                <Upload className="h-3.5 w-3.5" />
                Replace Image
              </button>
              <button
                type="button"
                onClick={handleLoadSample}
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs font-medium text-slate-200 transition hover:border-emerald-400"
              >
                <Sparkles className="h-3.5 w-3.5" />
                Load Sample
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

      {/* Right: Controls & detection breakdown */}
      <div className="space-y-4 lg:col-span-4">
        <div className="rounded-2xl border border-slate-800 bg-[#0F1623] p-5">
          <h4 className="font-mono text-xs font-semibold uppercase tracking-wider text-slate-400">
            NMS Decoding Parameters
          </h4>

          <div className="mt-5 space-y-5">
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

          <div className="mt-6 grid grid-cols-2 gap-3 border-t border-slate-800 pt-4">
            <div className="rounded-xl bg-slate-950/70 p-3">
              <p className="font-mono text-[11px] text-slate-400">LATENCY</p>
              <p className="mt-1 font-mono text-lg font-semibold text-white">
                {result ? `${result.inference_ms.toFixed(1)} ms` : "--"}
              </p>
            </div>
            <div className="rounded-xl bg-slate-950/70 p-3">
              <p className="font-mono text-[11px] text-slate-400">DETECTIONS</p>
              <p className="mt-1 font-mono text-lg font-semibold text-cyan-300">
                {result ? result.detections.length : 0}
              </p>
            </div>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-[#0F1623] p-5">
          <h4 className="font-mono text-xs font-semibold uppercase tracking-wider text-slate-400">
            Decoded Bounding Boxes
          </h4>
          <div className="mt-3 max-h-[340px] space-y-2 overflow-y-auto pr-1">
            {result && result.detections.length > 0 ? (
              result.detections.map((det, idx) => (
                <div
                  key={`${det.class_id}-${idx}`}
                  className="flex items-center justify-between rounded-xl border border-slate-800/90 bg-slate-950/70 px-3 py-2.5 text-xs"
                >
                  <div className="flex items-center gap-2.5">
                    <span
                      className="h-2.5 w-2.5 rounded-full"
                      style={{ backgroundColor: getClassColor(det.class_id) }}
                    />
                    <div>
                      <p className="font-semibold capitalize text-white">{det.class_name}</p>
                      <p className="font-mono text-[11px] text-slate-400">
                        [{Math.round(det.x1)}, {Math.round(det.y1)}, {Math.round(det.x2)},{" "}
                        {Math.round(det.y2)}]
                      </p>
                    </div>
                  </div>
                  <span className="font-mono font-semibold text-cyan-300">
                    {(det.score * 100).toFixed(1)}%
                  </span>
                </div>
              ))
            ) : (
              <p className="py-6 text-center text-xs text-slate-500">
                {selectedFile
                  ? "No detections above threshold."
                  : "Upload an image or load the sample scene to inspect detections."}
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

