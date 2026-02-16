import React, { useState } from 'react';
import { useNavigate, Link as RouterLink } from 'react-router-dom';
import {
  Box, Card, CardContent, Typography, TextField, Button,
  Alert, InputAdornment, IconButton, Link,
} from '@mui/material';
import {
  Visibility, VisibilityOff, LocalHospital as HospitalIcon,
  Login as LoginIcon,
} from '@mui/icons-material';
import { login } from '../services/api';
import { useAppStore } from '../store';

export default function LoginPage() {
  const navigate = useNavigate();
  const { setAuth } = useAppStore();
  const [form, setForm] = useState({ username: '', password: '' });
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.username || !form.password) {
      setError('Please fill in all fields.');
      return;
    }
    setLoading(true);
    setError('');
    try {
      const res = await login(form.username, form.password);
      setAuth(res.data.user, res.data.token);
      navigate('/');
    } catch (err: any) {
      const msg = err.response?.data?.error || err.response?.data?.detail || 'Login failed. Please check your credentials.';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'linear-gradient(135deg, #0d47a1 0%, #1565c0 50%, #1976d2 100%)',
      p: 2,
    }}>
      <Card sx={{ maxWidth: 440, width: '100%', borderRadius: 3 }}>
        <CardContent sx={{ p: { xs: 3, sm: 4 } }}>
          <Box sx={{ textAlign: 'center', mb: 3 }}>
            <HospitalIcon sx={{ fontSize: 48, color: 'primary.main', mb: 1 }} />
            <Typography variant="h5" fontWeight={700}>MS Risk Lab</Typography>
            <Typography variant="body2" color="text.secondary">
              AI Healthcare Interaction Embodiment
            </Typography>
          </Box>

          {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

          <form onSubmit={handleSubmit}>
            <TextField
              label="Username"
              value={form.username}
              onChange={(e) => setForm({ ...form, username: e.target.value })}
              fullWidth
              autoFocus
              sx={{ mb: 2 }}
            />
            <TextField
              label="Password"
              type={showPassword ? 'text' : 'password'}
              value={form.password}
              onChange={(e) => setForm({ ...form, password: e.target.value })}
              fullWidth
              sx={{ mb: 3 }}
              InputProps={{
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton onClick={() => setShowPassword(!showPassword)} edge="end">
                      {showPassword ? <VisibilityOff /> : <Visibility />}
                    </IconButton>
                  </InputAdornment>
                ),
              }}
            />
            <Button
              type="submit"
              variant="contained"
              fullWidth
              size="large"
              disabled={loading}
              startIcon={<LoginIcon />}
              sx={{ mb: 2, py: 1.5, borderRadius: 2 }}
            >
              {loading ? 'Signing in...' : 'Sign In'}
            </Button>
          </form>

          <Typography variant="body2" align="center" color="text.secondary">
            Don't have an account?{' '}
            <Link component={RouterLink} to="/register" underline="hover">
              Create one
            </Link>
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
}
