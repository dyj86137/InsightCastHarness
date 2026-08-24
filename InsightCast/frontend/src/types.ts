export type JsonObject = Record<string, any>;

export interface RunStatus {
  run_id: string;
  status: string;
  stage?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  discovered_count: number;
  accepted_count: number;
  pushed_count: number;
  error?: string | null;
  metadata: JsonObject;
}

export interface DashboardData {
  dashboard_date: string;
  discovered_count: number;
  pushed_count: number;
  pending_review_count: number;
  recent_runs: RunStatus[];
}

export interface Person {
  id: string;
  name: string;
  display_name?: string | null;
  aliases: string[];
  companies: string[];
  title?: string | null;
  industries: string[];
  importance: number;
  enabled: boolean;
  notes?: string | null;
  metadata: JsonObject;
  created_at: string;
  updated_at: string;
}

export interface Source {
  id: string;
  name: string;
  type: string;
  url?: string | null;
  platform_id?: string | null;
  authority: number;
  enabled: boolean;
  languages: string[];
  industries: string[];
  metadata: JsonObject;
  created_at: string;
  updated_at: string;
}

export interface InterviewSummary {
  summary: string;
  key_points: string[];
  potential_opportunities: string[];
  industry_judgements: string[];
  mentioned_companies: string[];
  mentioned_products: string[];
  audience: string[];
  novelty_assessment?: string | null;
  insights?: JsonObject[];
  [key: string]: any;
}

export interface Paginated<T> {
  items: T[];
  total: number;
  offset: number;
  limit: number;
}

export interface CandidateDecision {
  candidate_id: string;
  is_interview: boolean;
  target_person_present: boolean;
  content_type: string;
  is_original_or_authoritative_source: boolean;
  is_short_clip_or_commentary: boolean;
  confidence: number;
  reason: string;
  metadata: JsonObject;
  created_at: string;
}

export interface Candidate {
  id: string;
  source_id?: string | null;
  source_name?: string | null;
  source_type: string;
  platform_item_id?: string | null;
  title: string;
  description?: string | null;
  url: string;
  canonical_url?: string | null;
  format: string;
  published_at?: string | null;
  duration_seconds?: number | null;
  detected_person_ids: string[];
  detected_person_names: string[];
  query_id?: string | null;
  query_text?: string | null;
  status: string;
  raw_metadata: JsonObject;
  created_at: string;
  updated_at: string;
  interview_decision?: CandidateDecision | null;
}

export interface Interview {
  id: string;
  title: string;
  url: string;
  status: string;
  source_name?: string | null;
  person_names: string[];
  published_at?: string | null;
  duration_seconds?: number | null;
  transcript_status: string;
  summary?: InterviewSummary | null;
  transcript?: JsonObject | null;
  candidates?: Candidate[];
  candidate_count?: number;
  person_count?: number;
  [key: string]: any;
}

export interface TraceEvent {
  offset: number;
  event: {
    id?: string;
    run_id?: string;
    event_type?: string;
    step?: number;
    level?: string;
    message?: string;
    payload?: JsonObject;
    metadata?: JsonObject;
    created_at?: string;
    [key: string]: any;
  };
}

export interface TraceEvents {
  run_id: string;
  items: TraceEvent[];
  next_offset: number;
}

export interface BriefItem {
  interview_id: string;
  summary_id?: string | null;
  section: string;
  rank: number;
  title: string;
  url: string;
  person_names: string[];
  industries: string[];
  score: number;
  reason?: string | null;
}

export interface BriefData extends JsonObject {
  id: string;
  brief_date: string;
  title: string;
  status: string;
  items: BriefItem[];
  interviews?: Interview[];
}

export interface BriefSummary {
  id: string;
  brief_date: string;
  title: string;
  status: string;
  items: BriefItem[];
  updated_at?: string;
  created_at?: string;
}

export interface BriefResponse {
  brief_date: string;
  data: BriefData;
}
