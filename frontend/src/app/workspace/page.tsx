"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { ArrowLeft, Camera, Cpu, Image as ImageIcon, Scan, Video } from "lucide-react";
import CameraStreamMonitor from "@/components/CameraStreamMonitor";
import ImageMonitor from "@/components/ImageMonitor";
import VideoStreamMonitor from "@/components/VideoStreamMonitor";
import { fetchHealth } from "@/lib/api";
import type { HealthResponse, MonitorMode } from "@/lib/types";

const MONITORS: {
  id: MonitorMode;
  label: string;
  description: string;
  icon: typeof ImageIcon;
}[] = [
  {
    id: "image",
    label: "Process Image Monitor",
    description: "Static frame NMS inspection",
    icon: ImageIcon,
  },
  {
    id: "video",
    label: "Video Stream Monitor",
    description: "Local video with FPS throttle",
    icon: Video,
  },
  {
    id: "camera",
    label: "Camera Stream Monitor",
    description: "Live webcam telemetry",
    icon: Camera,
  },
];

function WorkspaceContent() {
  const searchParams = useSearchParams();
  const initialParam = searchParams.get("mode");
  const resolvedInitial: MonitorMode =
    initialParam === "video" || initialParam === "camera" ? initialParam : "image";

  const [activeMonitor, setActiveMonitor] = useState<MonitorMode>(resolvedInitial);
  const [confThreshold, setConfThreshold] = useState<number>(0.25);
  const [iouThreshold, setIouThreshold] = useState<number>(0.45);
  const [fps, setFps] = useState<number>(10);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<boolean>(false);

  useEffect(() => {
    if (initialParam === "image" || initialParam === "video" || initialParam === "camera") {
      setActiveMonitor(initialParam);
    }
  }, [initialParam]);

  useEffect(() => {
    let mounted = true;
    const pollHealth = async () => {
      try {
        const res = await fetchHealth();
        if (mounted) {
          setHealth(res);
          setHealthError(false);
        }
      } catch {
        if (mounted) {
          setHealthError(true);
        }
      }
    };
    void pollHealth();
    const timer = setInterval(pollHealth, 15000);
    return () => {
      mounted = false;
      clearInterval(timer);
    };
  }, []);

  return (
    <div className="min-h-screen bg-[#070B12] bg-telemetry-grid">
      {/* Sticky Top Header + Monitor Switcher */}
      <header className="sticky top-0 z-30 border-b border-slate-800/90 bg-[#070B12]/90 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-6 py-4">
          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="inline-flex items-center gap-2 rounded-lg border border-slate-800 bg-slate-900/80 px-3 py-1.5 text-xs font-medium text-slate-300 transition hover:border-cyan-400/60 hover:text-cyan-300"
            >
              <ArrowLeft className="h-3.5 w-3.5" />
              Landing
            </Link>

            <div className="flex items-center gap-2.5">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl border border-cyan-500/40 bg-cyan-500/10 text-cyan-300">
                <Scan className="h-4 w-4" />
              </div>
              <div>
                <h1 className="text-sm font-bold tracking-wide text-white">
                  YOLOv2 DETECTION WORKSPACE
                </h1>
                <p className="font-mono text-[11px] text-slate-400">
                  3-MONITOR REAL-TIME TELEMETRY
                </p>
              </div>
            </div>
          </div>

          {/* Backend Health Pill */}
          <div className="flex items-center gap-2 rounded-full border border-slate-800 bg-[#0F1623] px-3.5 py-1.5 font-mono text-xs">
            <span
              className={`h-2 w-2 rounded-full ${
                health?.model_loaded
                  ? "bg-emerald-400"
                  : healthError
                  ? "bg-rose-400"
                  : "bg-amber-400"
              }`}
            />
            <Cpu className="h-3.5 w-3.5 text-slate-400" />
            <span className="text-slate-200">
              {health?.model_loaded
                ? `FASTAPI READY · ${health.device.toUpperCase()} (${health.num_classes} CLS)`
                : healthError
                ? "BACKEND OFFLINE (PORT 8000)"
                : "CHECKING BACKEND..."}
            </span>
          </div>
        </div>

        {/* 3-Monitor Switcher Tabs */}
        <div className="mx-auto max-w-7xl px-6 pb-3">
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
            {MONITORS.map((monitor) => {
              const Icon = monitor.icon;
              const isActive = activeMonitor === monitor.id;
              return (
                <button
                  key={monitor.id}
                  type="button"
                  onClick={() => setActiveMonitor(monitor.id)}
                  className={`flex items-center gap-3 rounded-xl border px-4 py-2.5 text-left transition ${
                    isActive
                      ? "border-cyan-400/70 bg-cyan-500/10 text-white shadow-[0_0_20px_rgba(34,211,238,0.1)]"
                      : "border-slate-800 bg-[#0F1623]/80 text-slate-400 hover:border-slate-700 hover:text-slate-200"
                  }`}
                >
                  <Icon
                    className={`h-4 w-4 shrink-0 ${
                      isActive ? "text-cyan-300" : "text-slate-500"
                    }`}
                  />
                  <div className="min-w-0">
                    <p className="truncate text-sm font-semibold">{monitor.label}</p>
                    <p className="truncate text-[11px] text-slate-400">{monitor.description}</p>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      </header>

      {/* Active Monitor Content */}
      <main className="mx-auto max-w-7xl px-6 py-6">
        {activeMonitor === "image" && (
          <ImageMonitor
            confThreshold={confThreshold}
            iouThreshold={iouThreshold}
            onConfChange={setConfThreshold}
            onIouChange={setIouThreshold}
          />
        )}

        {activeMonitor === "video" && (
          <VideoStreamMonitor
            confThreshold={confThreshold}
            iouThreshold={iouThreshold}
            fps={fps}
            onConfChange={setConfThreshold}
            onIouChange={setIouThreshold}
            onFpsChange={setFps}
          />
        )}

        {activeMonitor === "camera" && (
          <CameraStreamMonitor
            confThreshold={confThreshold}
            iouThreshold={iouThreshold}
            fps={fps}
            onConfChange={setConfThreshold}
            onIouChange={setIouThreshold}
            onFpsChange={setFps}
          />
        )}
      </main>
    </div>
  );
}

export default function WorkspacePage() {
  return (
    <Suspense
      fallback={
        <div className="flex min-h-screen items-center justify-center bg-[#070B12] font-mono text-sm text-slate-400">
          Initializing YOLOv2 Workspace...
        </div>
      }
    >
      <WorkspaceContent />
    </Suspense>
  );
}

