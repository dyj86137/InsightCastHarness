import type {
  BriefResponse,
  BriefSummary,
  Candidate,
  DashboardData,
  Interview,
  JsonObject,
  Paginated,
  Person,
  RunStatus,
  Source,
  TraceEvents,
} from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8766").replace(/\/$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : detail;
    } catch {
      // Keep the HTTP status when the server did not return JSON.
    }
    throw new Error(detail);
  }

  if (response.status === 204) {
    return undefined as T;
  }
  return response.json() as Promise<T>;
}

function query(params: Record<string, string | number | undefined>): string {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== "") search.set(key, String(value));
  });
  const encoded = search.toString();
  return encoded ? `?${encoded}` : "";
}

export const api = {
  baseUrl: API_BASE,

  health: () => request<{ status: string }>("/health"),

  dashboard: (date: string) =>
    request<DashboardData>(`/api/dashboard${query({ dashboard_date: date })}`),

  runs: (params: { status?: string; limit?: number } = {}) =>
    request<Paginated<RunStatus>>(`/api/runs${query(params)}`),

  run: (id: string) =>
    request<RunStatus>(`/api/runs/${encodeURIComponent(id)}`),

  runEvents: (id: string, params: { offset?: number; limit?: number } = {}) =>
    request<TraceEvents>(`/api/runs/${encodeURIComponent(id)}/events${query(params)}`),

  createRun: () =>
    request<{ run_id: string; status: string }>("/api/runs", {
      method: "POST",
      body: JSON.stringify({}),
    }),

  candidates: (params: { status?: string; limit?: number } = {}) =>
    request<Paginated<Candidate>>(`/api/candidates${query(params)}`),

  candidate: (id: string) =>
    request<{ data: Candidate }>(`/api/candidates/${encodeURIComponent(id)}`),

  reviewCandidate: (
    id: string,
    action: "accept" | "reject" | "duplicate" | "archive",
    duplicateOfCandidateId?: string,
  ) =>
    request<{ data: Candidate }>(`/api/candidates/${encodeURIComponent(id)}/review`, {
      method: "POST",
      body: JSON.stringify({
        action,
        ...(duplicateOfCandidateId ? { duplicate_of_candidate_id: duplicateOfCandidateId } : {}),
      }),
    }),

  persons: (params: { enabled?: boolean; offset?: number; limit?: number } = {}) =>
    request<Paginated<Person>>(`/api/persons${query({ ...params, enabled: params.enabled === undefined ? undefined : String(params.enabled) })}`),

  person: (id: string) =>
    request<Person>(`/api/persons/${encodeURIComponent(id)}`),

  createPerson: (payload: Record<string, unknown>) =>
    request<Person>("/api/persons", { method: "POST", body: JSON.stringify(payload) }),

  updatePerson: (id: string, payload: Record<string, unknown>) =>
    request<Person>(`/api/persons/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify(payload) }),

  deletePerson: (id: string) =>
    request<void>(`/api/persons/${encodeURIComponent(id)}`, { method: "DELETE" }),

  sources: (params: { enabled?: boolean; offset?: number; limit?: number } = {}) =>
    request<Paginated<Source>>(`/api/sources${query({ ...params, enabled: params.enabled === undefined ? undefined : String(params.enabled) })}`),

  source: (id: string) =>
    request<Source>(`/api/sources/${encodeURIComponent(id)}`),

  createSource: (payload: Record<string, unknown>) =>
    request<Source>("/api/sources", { method: "POST", body: JSON.stringify(payload) }),

  updateSource: (id: string, payload: Record<string, unknown>) =>
    request<Source>(`/api/sources/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify(payload) }),

  deleteSource: (id: string) =>
    request<void>(`/api/sources/${encodeURIComponent(id)}`, { method: "DELETE" }),

  briefs: (params: { status?: string; limit?: number } = {}) =>
    request<Paginated<BriefSummary>>(`/api/briefs${query(params)}`),

  brief: (id: string) =>
    request<BriefResponse>(`/api/briefs/${encodeURIComponent(id)}`),

  interviews: (params: { status?: string; source_id?: string; person_id?: string; offset?: number; limit?: number } = {}) =>
    request<Paginated<Interview>>(`/api/interviews${query(params)}`),

  interview: (id: string) =>
    request<{ data: Interview }>(`/api/interviews/${encodeURIComponent(id)}`),
};

export function asObject(value: unknown): JsonObject {
  return value && typeof value === "object" ? (value as JsonObject) : {};
}
