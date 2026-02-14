import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box, Grid, Typography, Card, CardContent, Button, CircularProgress,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Paper, Chip, Alert,
} from '@mui/material';
import {
  People as PeopleIcon,
  Warning as WarningIcon,
  Assessment as AssessmentIcon,
  PlayArrow as PlayIcon,
  Speed as SpeedIcon,
  RateReview as ReviewIcon,
  AutoMode as AutoIcon,
  TrendingUp as TrendIcon,
} from '@mui/icons-material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RTooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend } from 'recharts';
import StatCard from '../components/StatCard';
import { useAppStore } from '../store';
import { getDashboard, triggerWorkflow } from '../services/api';
import { formatPercent, formatDate, getActionColor } from '../utils/helpers';
import type { DashboardData } from '../types';

const COLORS = ['#4caf50', '#1976d2', '#f57c00', '#d32f2f'];

export default function DashboardPage() {
  const navigate = useNavigate();
  const { dashboard, setDashboard, dashboardLoading, setDashboardLoading } = useAppStore();
  const [triggerLoading, setTriggerLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setDashboardLoading(true);
    setError(null);
    getDashboard()
      .then((r) => setDashboard(r.data))
      .catch((err) => setError(err.message || 'Failed to load dashboard data'))
      .finally(() => setDashboardLoading(false));
  }, [setDashboard, setDashboardLoading]);

  const handleTriggerWorkflow = async () => {
    setTriggerLoading(true);
    try {
      await triggerWorkflow();
      setTimeout(() => {
        getDashboard().then((r) => setDashboard(r.data)).catch(() => {});
        setTriggerLoading(false);
      }, 2000);
    } catch {
      setTriggerLoading(false);
    }
  };

  if (dashboardLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 10 }}>
        <CircularProgress size={60} />
      </Box>
    );
  }

  if (error || !dashboard) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 10 }}>
        <Alert severity="error" sx={{ maxWidth: 600 }}>
          {error || 'Unable to load dashboard. Please check that all services are running and refresh the page.'}
        </Alert>
      </Box>
    );
  }

  const actionData = dashboard.latest_run
    ? [
        { name: 'No Action', value: dashboard.latest_run.total_assessed - dashboard.latest_run.flagged, color: COLORS[0] },
        { name: 'Recommend', value: dashboard.latest_run.recommend_actions, color: COLORS[1] },
        { name: 'Draft Order', value: dashboard.latest_run.draft_actions, color: COLORS[2] },
        { name: 'Auto Order', value: dashboard.latest_run.auto_actions, color: COLORS[3] },
      ].filter(d => d.value > 0)
    : [];

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4">Dashboard</Typography>
          <Typography variant="body2" color="text.secondary">
            AI Healthcare Interaction Embodiment - MS Risk Screening
          </Typography>
        </Box>
        <Button
          variant="contained"
          size="large"
          startIcon={triggerLoading ? <CircularProgress size={20} color="inherit" /> : <PlayIcon />}
          onClick={handleTriggerWorkflow}
          disabled={triggerLoading}
          sx={{ borderRadius: 3, px: 4 }}
        >
          Run Screening Workflow
        </Button>
      </Box>

      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total Patients"
            value={dashboard.total_patients.toLocaleString()}
            subtitle="In EHR database"
            icon={<PeopleIcon />}
            color="#1565c0"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="At-Risk Population"
            value={dashboard.at_risk_count.toLocaleString()}
            subtitle={`${formatPercent(dashboard.at_risk_rate)} of total`}
            icon={<WarningIcon />}
            color="#d32f2f"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Pending Reviews"
            value={dashboard.pending_reviews}
            subtitle="Awaiting clinician review"
            icon={<ReviewIcon />}
            color="#f57c00"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Workflow Runs"
            value={dashboard.total_workflow_runs}
            subtitle="Total completed"
            icon={<AssessmentIcon />}
            color="#00897b"
          />
        </Grid>
      </Grid>

      {dashboard.latest_run && (
        <>
          <Grid container spacing={3} sx={{ mb: 3 }}>
            <Grid item xs={12} sm={6} md={3}>
              <StatCard
                title="Precision"
                value={formatPercent(dashboard.latest_run.precision)}
                subtitle="Latest run"
                icon={<TrendIcon />}
                color="#2e7d32"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <StatCard
                title="Recall"
                value={formatPercent(dashboard.latest_run.recall)}
                subtitle="Latest run"
                icon={<SpeedIcon />}
                color="#1565c0"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <StatCard
                title="Flagged Patients"
                value={dashboard.latest_run.flagged}
                subtitle={`of ${dashboard.latest_run.total_assessed} assessed`}
                icon={<WarningIcon />}
                color="#f57c00"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <StatCard
                title="Auto Actions"
                value={dashboard.latest_run.auto_actions}
                subtitle="Autonomous orders"
                icon={<AutoIcon />}
                color="#d32f2f"
              />
            </Grid>
          </Grid>

          <Grid container spacing={3} sx={{ mb: 3 }}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>Action Distribution</Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={actionData}
                        cx="50%"
                        cy="50%"
                        outerRadius={100}
                        dataKey="value"
                        label={({ name, value }) => `${name}: ${value}`}
                      >
                        {actionData.map((entry, i) => (
                          <Cell key={i} fill={entry.color} />
                        ))}
                      </Pie>
                      <Legend />
                      <RTooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>Autonomy Levels</Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={[
                      { name: 'Recommend Only', count: dashboard.latest_run.recommend_actions, fill: '#1976d2' },
                      { name: 'Draft Order', count: dashboard.latest_run.draft_actions, fill: '#f57c00' },
                      { name: 'Auto Order', count: dashboard.latest_run.auto_actions, fill: '#d32f2f' },
                    ]}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <RTooltip />
                      <Bar dataKey="count" fill="#1976d2">
                        {[0, 1, 2].map(i => (
                          <Cell key={i} fill={['#1976d2', '#f57c00', '#d32f2f'][i]} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </>
      )}

      {!dashboard.latest_run && (
        <Alert severity="info" sx={{ mb: 3, borderRadius: 2 }}>
          No workflow runs yet. Click "Run Screening Workflow" to start the multi-agent MS risk screening pipeline.
        </Alert>
      )}

      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>Recent Workflow Runs</Typography>
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Run ID</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Policy</TableCell>
                  <TableCell align="right">Candidates</TableCell>
                  <TableCell align="right">Flagged</TableCell>
                  <TableCell align="right">Precision</TableCell>
                  <TableCell align="right">Recall</TableCell>
                  <TableCell>Date</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {dashboard.recent_runs.map((run) => (
                  <TableRow
                    key={run.id}
                    hover
                    sx={{ cursor: 'pointer' }}
                    onClick={() => navigate(`/workflows/${run.id}`)}
                  >
                    <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
                      {run.id.slice(0, 8)}...
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={run.status}
                        size="small"
                        color={run.status === 'COMPLETED' ? 'success' : run.status === 'FAILED' ? 'error' : 'warning'}
                      />
                    </TableCell>
                    <TableCell>{run.policy_name || 'Default'}</TableCell>
                    <TableCell align="right">{run.candidates_found}</TableCell>
                    <TableCell align="right">{run.flagged_count}</TableCell>
                    <TableCell align="right">{formatPercent(run.precision)}</TableCell>
                    <TableCell align="right">{formatPercent(run.recall)}</TableCell>
                    <TableCell>{formatDate(run.created_at)}</TableCell>
                  </TableRow>
                ))}
                {dashboard.recent_runs.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={8} align="center" sx={{ py: 4 }}>
                      No workflow runs yet
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>
    </Box>
  );
}
