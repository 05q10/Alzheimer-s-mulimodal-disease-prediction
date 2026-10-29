/* ArchitectureDiagram.tsx — SVG-based LSTM architecture visualization */
"use client";

import { motion } from "framer-motion";

const layerVariant = {
  hidden: { opacity: 0, y: 20 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.15, duration: 0.5, ease: "easeOut" as const },
  }),
};

function LayerBox({
  x,
  y,
  width,
  height,
  label,
  sub,
  color,
  index,
}: {
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
  sub: string;
  color: string;
  index: number;
}) {
  return (
    <motion.g custom={index} variants={layerVariant} initial="hidden" animate="visible">
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        rx={10}
        fill={color}
        stroke="rgba(255,255,255,0.15)"
        strokeWidth={1}
      />
      <text
        x={x + width / 2}
        y={y + height / 2 - 6}
        textAnchor="middle"
        fill="white"
        fontSize={13}
        fontWeight={600}
      >
        {label}
      </text>
      <text
        x={x + width / 2}
        y={y + height / 2 + 12}
        textAnchor="middle"
        fill="rgba(255,255,255,0.6)"
        fontSize={10}
      >
        {sub}
      </text>
    </motion.g>
  );
}

function Arrow({ x1, y1, x2, y2, index }: { x1: number; y1: number; x2: number; y2: number; index: number }) {
  return (
    <motion.line
      x1={x1}
      y1={y1}
      x2={x2}
      y2={y2}
      stroke="rgba(96,165,250,0.5)"
      strokeWidth={2}
      markerEnd="url(#arrowhead)"
      custom={index}
      variants={layerVariant}
      initial="hidden"
      animate="visible"
    />
  );
}

export default function ArchitectureDiagram() {
  const cx = 300;
  const bw = 220;
  const bh = 52;
  const gap = 22;
  const startY = 20;

  const layers = [
    { label: "Input Layer", sub: "3 timesteps × 5 features", color: "rgba(59,130,246,0.25)" },
    { label: "Bidirectional LSTM", sub: "64 units → 128 output", color: "rgba(139,92,246,0.3)" },
    { label: "Batch Normalization", sub: "128 features", color: "rgba(16,185,129,0.25)" },
    { label: "Dense (ReLU)", sub: "32 units · L2 reg", color: "rgba(245,158,11,0.25)" },
    { label: "Dropout", sub: "rate = 0.3", color: "rgba(107,114,128,0.25)" },
    { label: "Dense (Softmax)", sub: "3 classes → CN / MCI / AD", color: "rgba(239,68,68,0.25)" },
  ];

  return (
    <div className="w-full flex justify-center overflow-x-auto py-4">
      <svg viewBox="0 0 600 500" width="100%" className="max-w-lg">
        <defs>
          <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="rgba(96,165,250,0.6)" />
          </marker>
        </defs>

        {layers.map((l, i) => {
          const ly = startY + i * (bh + gap);
          return (
            <g key={i}>
              <LayerBox
                x={cx - bw / 2}
                y={ly}
                width={bw}
                height={bh}
                label={l.label}
                sub={l.sub}
                color={l.color}
                index={i}
              />
              {i < layers.length - 1 && (
                <Arrow x1={cx} y1={ly + bh} x2={cx} y2={ly + bh + gap} index={i} />
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
