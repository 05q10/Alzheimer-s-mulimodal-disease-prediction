"use client";

import React from "react";
import { TwinFull, AddVisitResponse } from "@/app/lib/api";
import FullFollowUpVisitForm from "./FullFollowUpVisitForm";

interface AddVisitPanelProps {
  twin: TwinFull;
  onVisitAdded: (updatedTwin: TwinFull, diff: AddVisitResponse["prob_diff"]) => void;
}

export default function AddVisitPanel({ twin, onVisitAdded }: AddVisitPanelProps) {
  return (
    <FullFollowUpVisitForm
      twin={twin}
      onVisitAdded={(updated) => onVisitAdded(updated, { CN: 0, MCI: 0, AD: 0 })}
    />
  );
}
