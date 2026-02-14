import axios from 'axios';
import type {
  Patient, RiskAssessment, PolicyConfiguration, WorkflowRun,
  DashboardData, SubgroupData, RiskDistribution, CalibrationPoint,
  WhatIfResult, Notification, AuditLog, GovernanceRule,
  MCPTool, A2AAgent, PaginatedResponse,
} from '../types';

const API_BASE = process.env.REACT_APP_API_URL || 'http://108.48.39.238:4055';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

// Patients
export const getPatients = (params?: Record<string, any>) =>
  api.get<PaginatedResponse<Patient>>('/api/patients/', { params });

export const getPatient = (id: string) =>
  api.get<Patient>(`/api/patients/${id}/`);

export const getPatientSummary = () =>
  api.get('/api/patients/summary/');

export const getPatientRiskHistory = (id: string) =>
  api.get<RiskAssessment[]>(`/api/patients/${id}/risk_history/`);

// Risk Assessments
export const getAssessments = (params?: Record<string, any>) =>
  api.get<PaginatedResponse<RiskAssessment>>('/api/assessments/', { params });

export const getAssessment = (id: string) =>
  api.get<RiskAssessment>(`/api/assessments/${id}/`);

export const getHighRiskAssessments = (params?: Record<string, any>) =>
  api.get<RiskAssessment[]>('/api/assessments/high_risk/', { params });

export const getPendingReviews = () =>
  api.get<RiskAssessment[]>('/api/assessments/pending_review/');

export const reviewAssessment = (id: string, data: {
  reviewed_by: string;
  review_notes: string;
  override_action?: string;
}) =>
  api.post<RiskAssessment>(`/api/assessments/${id}/review/`, data);

// Policies
export const getPolicies = () =>
  api.get<PaginatedResponse<PolicyConfiguration>>('/api/policies/');

export const createPolicy = (data: Partial<PolicyConfiguration>) =>
  api.post<PolicyConfiguration>('/api/policies/', data);

export const activatePolicy = (id: string) =>
  api.post<PolicyConfiguration>(`/api/policies/${id}/activate/`);

// Workflows
export const getWorkflowRuns = () =>
  api.get<PaginatedResponse<WorkflowRun>>('/api/workflows/');

export const getWorkflowRun = (id: string) =>
  api.get<WorkflowRun>(`/api/workflows/${id}/`);

export const triggerWorkflow = (data?: { policy_id?: string; patient_limit?: number }) =>
  api.post('/api/workflows/trigger/', data || {});

export const getWorkflowMetrics = (id: string) =>
  api.get('/api/workflows/' + id + '/metrics/');

export const getWorkflowRiskDistribution = (id: string) =>
  api.get<RiskDistribution>('/api/workflows/' + id + '/risk_distribution/');

export const getWorkflowActionDistribution = (id: string) =>
  api.get('/api/workflows/' + id + '/action_distribution/');

export const getWorkflowAutonomyDistribution = (id: string) =>
  api.get('/api/workflows/' + id + '/autonomy_distribution/');

export const getWorkflowCalibration = (id: string) =>
  api.get<CalibrationPoint[]>('/api/workflows/' + id + '/calibration/');

export const getWorkflowFairness = (id: string, groupBy: string) =>
  api.get<SubgroupData[]>('/api/workflows/' + id + '/fairness/', { params: { group_by: groupBy } });

// What-If Analysis
export const runWhatIf = (data: {
  run_id: string;
  risk_review_threshold?: number;
  draft_order_threshold?: number;
  auto_order_threshold?: number;
  max_auto_actions_per_day?: number;
}) =>
  api.post<WhatIfResult>('/api/what-if/', data);

// Dashboard
export const getDashboard = () =>
  api.get<DashboardData>('/api/dashboard/');

// Governance
export const getGovernanceRules = () =>
  api.get<PaginatedResponse<GovernanceRule>>('/api/governance-rules/');

export const getComplianceReports = () =>
  api.get('/api/compliance-reports/');

export const generateComplianceReport = (runId: string) =>
  api.post('/api/compliance-reports/generate/', { run_id: runId });

// Audit Logs
export const getAuditLogs = (params?: Record<string, any>) =>
  api.get<PaginatedResponse<AuditLog>>('/api/audit-logs/', { params });

// Notifications
export const getNotifications = () =>
  api.get<PaginatedResponse<Notification>>('/api/notifications/');

export const markNotificationRead = (id: string) =>
  api.post(`/api/notifications/${id}/mark_read/`);

export const markAllNotificationsRead = () =>
  api.post('/api/notifications/mark_all_read/');

export const getUnreadCount = () =>
  api.get<{ unread_count: number }>('/api/notifications/unread_count/');

// MCP
export const getMCPTools = () =>
  api.get<{ tools: MCPTool[] }>('/mcp/tools');

export const invokeMCPTool = (sessionId: string, toolName: string, args: Record<string, any>) =>
  api.post('/mcp/invoke', { session_id: sessionId, tool_name: toolName, arguments: args });

export const createMCPSession = () =>
  api.post<{ session_id: string }>('/mcp/session');

// A2A
export const getA2AAgents = () =>
  api.get<{ agents: A2AAgent[] }>('/a2a/agents');

export const createA2ATask = (data: {
  to_agent: string;
  action: string;
  payload: Record<string, any>;
}) =>
  api.post('/a2a/tasks/create', { from_agent: 'user', ...data });

export const executeA2ATask = (taskId: string) =>
  api.post(`/a2a/tasks/${taskId}/execute`);

export const orchestrateScreening = (patientId: string) =>
  api.post('/a2a/orchestrate', { patient_id: patientId });

export default api;
