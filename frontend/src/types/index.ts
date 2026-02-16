export interface User {
  id: number;
  username: string;
  email: string;
  first_name: string;
  last_name: string;
  date_joined: string;
  is_staff: boolean;
}

export interface AuthResponse {
  token: string;
  user: User;
}

export interface Patient {
  id: string;
  patient_id: string;
  age: number;
  sex: string;
  visits_last_year: number;
  lookalike_dx: string;
  note: string;
  has_mri: boolean;
  mri_lesions: boolean;
  note_has_ms_terms: boolean;
  true_at_risk: boolean;
  symptom_count: number;
  optic_neuritis: boolean;
  paresthesia: boolean;
  weakness: boolean;
  gait_instability: boolean;
  vertigo: boolean;
  fatigue: boolean;
  bladder_issues: boolean;
  cognitive_fog: boolean;
  vitamin_d_ngml: number | null;
  vitamin_d_deficient: boolean | null;
  infectious_mono_history: boolean | null;
  smartform_neuro_symptom_score: number | null;
  paths_like_function_score: number | null;
  created_at: string;
}

export interface RiskAssessment {
  id: string;
  patient: string;
  patient_display: string;
  run_id: string;
  risk_score: number;
  action: ActionType;
  autonomy_level: AutonomyLevel;
  feature_contributions: Record<string, number>;
  flags: string[];
  flag_count: number;
  rationale: string[];
  notes_analysis: Record<string, any>;
  llm_summary: string;
  patient_card: Record<string, any>;
  reviewed_by: string;
  review_notes: string;
  reviewed_at: string | null;
  created_at: string;
}

export type ActionType =
  | 'NO_ACTION'
  | 'RECOMMEND_NEURO_REVIEW'
  | 'DRAFT_MRI_ORDER'
  | 'AUTO_ORDER_MRI_AND_NOTIFY_NEURO';

export type AutonomyLevel =
  | 'RECOMMEND_ONLY'
  | 'DRAFT_ORDER'
  | 'AUTO_ORDER_WITH_GUARDRAILS';

export interface PolicyConfiguration {
  id: string;
  name: string;
  risk_review_threshold: number;
  draft_order_threshold: number;
  auto_order_threshold: number;
  max_auto_actions_per_day: number;
  is_active: boolean;
  created_by: string;
  created_at: string;
  updated_at: string;
}

export interface WorkflowRun {
  id: string;
  policy: string;
  policy_name: string;
  status: 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED';
  total_patients: number;
  candidates_found: number;
  flagged_count: number;
  precision: number | null;
  recall: number | null;
  auto_actions: number;
  draft_actions: number;
  recommend_actions: number;
  safety_flag_rate: number | null;
  duration_seconds: number | null;
  error_message: string;
  created_at: string;
  updated_at: string;
}

export interface DashboardData {
  total_patients: number;
  at_risk_count: number;
  at_risk_rate: number;
  latest_run: {
    run_id: string;
    status: string;
    total_assessed: number;
    flagged: number;
    precision: number | null;
    recall: number | null;
    auto_actions: number;
    draft_actions: number;
    recommend_actions: number;
    created_at: string;
  } | null;
  pending_reviews: number;
  unread_notifications: number;
  recent_runs: WorkflowRun[];
  total_workflow_runs: number;
}

export interface SubgroupData {
  group: string;
  n: number;
  flagged_rate: number;
  avg_risk: number;
  auto_or_draft_rate: number;
  auto_rate: number;
  safety_flag_rate: number;
  true_at_risk_rate: number;
  mri_rate: number;
}

export interface RiskDistribution {
  bins: number[];
  counts: number[];
  stats: {
    mean: number;
    median: number;
    std: number;
    min: number;
    max: number;
    q25: number;
    q75: number;
  };
}

export interface CalibrationPoint {
  bin: number;
  mean_predicted: number;
  mean_actual: number;
  count: number;
}

export interface WhatIfResult {
  policy: Record<string, number>;
  results: {
    flagged: number;
    auto: number;
    draft: number;
    recommend: number;
    no_action: number;
  };
  precision: number;
  recall: number;
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

export interface Notification {
  id: string;
  title: string;
  message: string;
  severity: 'info' | 'warning' | 'critical' | 'success';
  category: string;
  is_read: boolean;
  related_patient_id: string;
  metadata: Record<string, any>;
  created_at: string;
}

export interface AuditLog {
  id: string;
  action_type: string;
  actor: string;
  target_type: string;
  target_id: string;
  details: Record<string, any>;
  ip_address: string;
  created_at: string;
}

export interface GovernanceRule {
  id: string;
  name: string;
  description: string;
  rule_type: string;
  condition: Record<string, any>;
  severity: string;
  is_active: boolean;
  created_at: string;
}

export interface MCPTool {
  name: string;
  description: string;
  input_schema: Record<string, any>;
}

export interface A2AAgent {
  agent_id: string;
  name: string;
  description: string;
  capabilities: string[];
  version: string;
}

export interface PaginatedResponse<T> {
  count: number;
  next: string | null;
  previous: string | null;
  results: T[];
}
