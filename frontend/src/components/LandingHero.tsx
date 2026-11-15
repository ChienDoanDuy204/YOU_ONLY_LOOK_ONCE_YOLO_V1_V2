"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  Activity,
  ArrowRight,
  Camera,
  Cpu,
  Image as ImageIcon,
  Layers,
  Scan,
  Video,
} from "lucide-react";

interface DemoDetection {
  id: string;
  label: string;
  confidence: string;
  color: string;
  borderClass: string;
  bgClass: string;
  badgeClass: string;
  left: string;
  top: string;
  width: string;
  height: string;
  revealAt: number;
}

const DEMO_BOXES: DemoDetection[] = [
  {
    id: "horse",
    label: "horse",
    confidence: "94.2%",
    color: "#22D3EE",
    borderClass: "border-cyan-400",
    bgClass: "bg-cyan-400/10",
    badgeClass: "bg-cyan-400 text-slate-950",
    left: "10.6%",
    top: "19.7%",
    width: "83.6%",
    height: "75.3%",
    revealAt: 1,
  },
  {
    id: "person",
    label: "person",
    confidence: "89.6%",
    color: "#34D399",
    borderClass: "border-emerald-400",
    bgClass: "bg-emerald-400/10",
    badgeClass: "bg-emerald-400 text-slate-950",
    left: "31.6%",
    top: "10.0%",
    width: "26.2%",
    height: "27.8%",
    revealAt: 2,
  },
];

const PIPELINE_STAGES = [
  "Scanning 416×416 RGB tensor...",
  "Decoding 13×13×5 anchor grid & horse proposal...",
  "Applying class-aware NMS (IoU 0.45)...",
  "2 objects locked · VOC confidence verified",
];

export default function LandingHero() {
  const [stage, setStage] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setStage((prev) => (prev + 1) % 4);
    }, 1500);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="relative min-h-screen overflow-hidden bg-[#070B12] bg-telemetry-grid">
      {/* Ambient radial lighting */}
      <div className="pointer-events-none absolute -top-40 left-1/4 h-96 w-96 rounded-full bg-cyan-500/10 blur-3xl" />
      <div className="pointer-events-none absolute bottom-0 right-1/4 h-96 w-96 rounded-full bg-emerald-500/10 blur-3xl" />

      {/* Top header */}
      <header className="relative z-10 mx-auto flex max-w-7xl items-center justify-between px-6 py-6">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-cyan-500/40 bg-cyan-500/10 text-cyan-300">
            <Scan className="h-5 w-5" />
          </div>
          <div>
            <p className="text-sm font-semibold tracking-wide text-white">YOLOv2 VISION STUDIO</p>
            <p className="font-mono text-xs text-slate-400">RESNET-18 · 5 KMEANS ANCHORS · 20 VOC CLASSES</p>
          </div>
        </div>

        <Link
          href="/workspace"
          className="inline-flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-900/80 px-4 py-2 text-sm font-medium text-slate-200 transition hover:border-cyan-400/60 hover:text-cyan-300"
        >
          Open Workspace
          <ArrowRight className="h-4 w-4" />
        </Link>
      </header>

      {/* Main split hero */}
      <main className="relative z-10 mx-auto grid max-w-7xl grid-cols-1 items-center gap-12 px-6 pb-20 pt-8 lg:grid-cols-12">
        {/* Left copy + Start CTA */}
        <div className="space-y-8 lg:col-span-6">
          <div className="inline-flex items-center gap-2 rounded-full border border-cyan-500/30 bg-cyan-500/10 px-3.5 py-1.5 text-xs font-medium text-cyan-300">
            <span className="h-2 w-2 animate-pulse rounded-full bg-cyan-400" />
            Real-Time FastAPI + Next.js Multi-Monitor Pipeline
          </div>

          <div className="space-y-4">
            <h1 className="text-4xl font-bold tracking-tight text-white sm:text-5xl lg:text-6xl">
              Single-Shot Detection,{" "}
              <span className="bg-gradient-to-r from-cyan-300 via-sky-400 to-emerald-300 bg-clip-text text-transparent">
                Live Telemetry.
              </span>
            </h1>
            <p className="max-w-xl text-base leading-relaxed text-slate-300 sm:text-lg">
              Inspect trained YOLOv2 bounding boxes with configurable{" "}
              <span className="font-mono text-cyan-300">Confidence</span> &{" "}
              <span className="font-mono text-emerald-300">IoU NMS</span> filtering across
              single images, local video streams, and live camera feeds with dynamic FPS control.
            </p>
          </div>

          {/* Primary Start Button */}
          <div className="flex flex-wrap items-center gap-4">
            <Link
              href="/workspace"
              className="group inline-flex items-center gap-3 rounded-xl bg-gradient-to-r from-cyan-400 to-emerald-400 px-7 py-4 text-base font-semibold text-slate-950 shadow-lg shadow-cyan-500/25 transition hover:scale-[1.02] hover:brightness-110 active:scale-[0.99]"
            >
              Start Detection
              <ArrowRight className="h-5 w-5 transition-transform group-hover:translate-x-1" />
            </Link>

            <div className="flex items-center gap-2 rounded-xl border border-slate-800 bg-slate-900/70 px-4 py-3 font-mono text-xs text-slate-300">
              <Cpu className="h-4 w-4 text-cyan-400" />
              <span>PyTorch · FastAPI · WebSocket NMS</span>
            </div>
          </div>

          {/* 3 Monitor quick cards */}
          <div className="grid grid-cols-1 gap-3 pt-2 sm:grid-cols-3">
            <Link
              href="/workspace?mode=image"
              className="group rounded-xl border border-slate-800 bg-[#0F1623]/90 p-4 transition hover:border-cyan-500/50"
            >
              <ImageIcon className="mb-2 h-5 w-5 text-cyan-400 transition group-hover:scale-110" />
              <p className="text-sm font-semibold text-white">Image Monitor</p>
              <p className="mt-1 text-xs text-slate-400">Upload stills & tune Confidence / IoU</p>
            </Link>

            <Link
              href="/workspace?mode=video"
              className="group rounded-xl border border-slate-800 bg-[#0F1623]/90 p-4 transition hover:border-emerald-500/50"
            >
              <Video className="mb-2 h-5 w-5 text-emerald-400 transition group-hover:scale-110" />
              <p className="text-sm font-semibold text-white">Video Stream</p>
              <p className="mt-1 text-xs text-slate-400">Play video clips with 1–30 FPS control</p>
            </Link>

            <Link
              href="/workspace?mode=camera"
              className="group rounded-xl border border-slate-800 bg-[#0F1623]/90 p-4 transition hover:border-amber-500/50"
            >
              <Camera className="mb-2 h-5 w-5 text-amber-400 transition group-hover:scale-110" />
              <p className="text-sm font-semibold text-white">Camera Stream</p>
              <p className="mt-1 text-xs text-slate-400">Live webcam capture with FPS throttle</p>
            </Link>
          </div>
        </div>

        {/* Right animated detection showcase */}
        <div className="lg:col-span-6">
          <div className="relative rounded-2xl border border-slate-800 bg-[#0F1623]/95 p-4 shadow-2xl shadow-black/60">
            {/* Window chrome */}
            <div className="mb-3 flex items-center justify-between border-b border-slate-800/80 pb-3">
              <div className="flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-full bg-rose-500/80" />
                <span className="h-2.5 w-2.5 rounded-full bg-amber-400/80" />
                <span className="h-2.5 w-2.5 rounded-full bg-emerald-400/80" />
                <span className="ml-2 font-mono text-xs text-slate-400">
                  LIVE_DEMO · 2008_000008.jpg
                </span>
              </div>
              <div className="flex items-center gap-2 font-mono text-xs text-cyan-300">
                <Activity className="h-3.5 w-3.5 animate-pulse" />
                <span>NMS ACTIVE</span>
              </div>
            </div>

            {/* Image viewport with laser sweep + animated bounding boxes */}
            <div className="relative aspect-[500/442] w-full overflow-hidden rounded-xl border border-slate-800 bg-slate-950">
              <img
                src="/demo-scene.jpg"
                alt="YOLOv2 detection showcase sample"
                className="h-full w-full object-cover"
              />

              {/* Subtle 13x13 anchor grid overlay */}
              <div
                className={`pointer-events-none absolute inset-0 transition-opacity duration-500 ${
                  stage === 0 || stage === 1 ? "opacity-30" : "opacity-10"
                }`}
                style={{
                  backgroundImage:
                    "linear-gradient(to right, rgba(34,211,238,0.28) 1px, transparent 1px), linear-gradient(to bottom, rgba(34,211,238,0.28) 1px, transparent 1px)",
                  backgroundSize: "calc(100% / 13) calc(100% / 13)",
                }}
              />

              {/* Animated laser scan sweep */}
              <div className="scan-sweep-line pointer-events-none absolute inset-x-0 top-0 h-16 bg-gradient-to-b from-transparent via-cyan-400/25 to-cyan-300/60">
                <div className="absolute inset-x-0 bottom-0 h-[2px] bg-cyan-300 shadow-[0_0_14px_#22D3EE]" />
              </div>

              {/* Staggered bounding box reveals */}
              {DEMO_BOXES.map((box) => {
                const visible = stage >= box.revealAt;
                return (
                  <div
                    key={box.id}
                    style={{
                      left: box.left,
                      top: box.top,
                      width: box.width,
                      height: box.height,
                    }}
                    className={`pointer-events-none absolute border-2 transition-all duration-500 ${box.borderClass} ${box.bgClass} ${
                      visible ? "scale-100 opacity-100" : "scale-95 opacity-0"
                    }`}
                  >
                    <div
                      className={`-mt-6 inline-flex items-center gap-1.5 rounded-t px-2 py-0.5 font-mono text-xs font-bold shadow ${box.badgeClass}`}
                    >
                      <span>{box.label}</span>
                      <span>{box.confidence}</span>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Bottom telemetry readout bar */}
            <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
              <div className="rounded-lg border border-slate-800 bg-slate-950/80 px-3 py-2.5 sm:col-span-2">
                <div className="flex items-center gap-2 font-mono text-xs text-slate-400">
                  <Layers className="h-3.5 w-3.5 text-cyan-400" />
                  <span>PIPELINE TRACE</span>
                </div>
                <p className="mt-1 font-mono text-xs text-cyan-200">{PIPELINE_STAGES[stage]}</p>
              </div>
              <div className="rounded-lg border border-slate-800 bg-slate-950/80 px-3 py-2.5">
                <p className="font-mono text-xs text-slate-400">ANCHOR GRID</p>
                <p className="mt-1 font-mono text-sm font-semibold text-emerald-300">
                  13×13 × 5
                </p>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

