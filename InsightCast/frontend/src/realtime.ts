import { io, type Socket } from "socket.io-client";

import { api } from "./api";

export interface IndustryUpdate {
  run_id: string;
  brief_id?: string | null;
  brief_date?: string | null;
  interview_id: string;
  title: string;
  url: string;
  section?: string | null;
  rank?: number | null;
  reason?: string | null;
  source_name?: string | null;
  person_names: string[];
  industries: string[];
  summary?: string | null;
  key_points: string[];
  mentioned_companies: string[];
  mentioned_products: string[];
}

const SOCKET_URL = (
  import.meta.env.VITE_SOCKET_URL ?? api.baseUrl
).replace(/\/$/, "");

export function createRealtimeSocket(): Socket {
  return io(SOCKET_URL, {
    path: "/socket.io",
    transports: ["websocket", "polling"],
    tryAllTransports: true,
    reconnection: true,
  });
}

export function normalizeKeywords(values: string[]): string[] {
  const seen = new Set<string>();
  const keywords: string[] = [];
  for (const value of values) {
    const keyword = value.trim().replace(/\s+/g, " ").toLocaleLowerCase();
    if (!keyword || keyword.length > 80 || seen.has(keyword)) continue;
    keywords.push(keyword);
    seen.add(keyword);
    if (keywords.length === 20) break;
  }
  return keywords;
}
