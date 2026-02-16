import React, { useMemo } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import { SnackbarProvider } from 'notistack';
import { useAppStore } from './store';
import Layout from './components/Layout';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import ProfilePage from './pages/ProfilePage';
import DashboardPage from './pages/DashboardPage';
import PatientsPage from './pages/PatientsPage';
import PatientDetailPage from './pages/PatientDetailPage';
import AssessmentsPage from './pages/AssessmentsPage';
import WorkflowsPage from './pages/WorkflowsPage';
import WorkflowDetailPage from './pages/WorkflowDetailPage';
import FairnessPage from './pages/FairnessPage';
import WhatIfPage from './pages/WhatIfPage';
import PoliciesPage from './pages/PoliciesPage';
import GovernancePage from './pages/GovernancePage';
import AgentsPage from './pages/AgentsPage';
import AuditPage from './pages/AuditPage';
import NotificationsPage from './pages/NotificationsPage';

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const isAuthenticated = useAppStore((s) => s.isAuthenticated);
  if (!isAuthenticated) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

function App() {
  const darkMode = useAppStore((s) => s.darkMode);
  const isAuthenticated = useAppStore((s) => s.isAuthenticated);

  const theme = useMemo(
    () =>
      createTheme({
        palette: {
          mode: darkMode ? 'dark' : 'light',
          primary: { main: '#1565c0' },
          secondary: { main: '#00897b' },
          error: { main: '#d32f2f' },
          warning: { main: '#f57c00' },
          success: { main: '#2e7d32' },
          background: darkMode
            ? { default: '#0a1929', paper: '#132f4c' }
            : { default: '#f5f7fa', paper: '#ffffff' },
        },
        typography: {
          fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
          h4: { fontWeight: 700 },
          h5: { fontWeight: 600 },
          h6: { fontWeight: 600 },
        },
        shape: { borderRadius: 12 },
        components: {
          MuiCard: {
            styleOverrides: {
              root: {
                boxShadow: darkMode
                  ? '0 2px 12px rgba(0,0,0,0.3)'
                  : '0 2px 12px rgba(0,0,0,0.08)',
              },
            },
          },
        },
      }),
    [darkMode]
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <SnackbarProvider maxSnack={3} anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}>
        <Router>
          <Routes>
            <Route path="/login" element={isAuthenticated ? <Navigate to="/" replace /> : <LoginPage />} />
            <Route path="/register" element={isAuthenticated ? <Navigate to="/" replace /> : <RegisterPage />} />
            <Route path="/*" element={
              <ProtectedRoute>
                <Layout>
                  <Routes>
                    <Route path="/" element={<DashboardPage />} />
                    <Route path="/patients" element={<PatientsPage />} />
                    <Route path="/patients/:id" element={<PatientDetailPage />} />
                    <Route path="/assessments" element={<AssessmentsPage />} />
                    <Route path="/workflows" element={<WorkflowsPage />} />
                    <Route path="/workflows/:id" element={<WorkflowDetailPage />} />
                    <Route path="/fairness" element={<FairnessPage />} />
                    <Route path="/what-if" element={<WhatIfPage />} />
                    <Route path="/policies" element={<PoliciesPage />} />
                    <Route path="/governance" element={<GovernancePage />} />
                    <Route path="/agents" element={<AgentsPage />} />
                    <Route path="/audit" element={<AuditPage />} />
                    <Route path="/notifications" element={<NotificationsPage />} />
                    <Route path="/profile" element={<ProfilePage />} />
                  </Routes>
                </Layout>
              </ProtectedRoute>
            } />
          </Routes>
        </Router>
      </SnackbarProvider>
    </ThemeProvider>
  );
}

export default App;
