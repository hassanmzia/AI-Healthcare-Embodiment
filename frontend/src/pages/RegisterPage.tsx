import React, { useState } from 'react';
import { useNavigate, Link as RouterLink } from 'react-router-dom';
import {
  Box, Card, CardContent, Typography, TextField, Button,
  Alert, InputAdornment, IconButton, Link, Grid,
} from '@mui/material';
import {
  Visibility, VisibilityOff, LocalHospital as HospitalIcon,
  PersonAdd as RegisterIcon,
} from '@mui/icons-material';
import { register } from '../services/api';
import { useAppStore } from '../store';

export default function RegisterPage() {
  const navigate = useNavigate();
  const { setAuth } = useAppStore();
  const [form, setForm] = useState({
    username: '', email: '', password: '', password_confirm: '',
    first_name: '', last_name: '',
  });
  const [showPassword, setShowPassword] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.username || !form.password || !form.password_confirm) {
      setErrors({ general: 'Please fill in all required fields.' });
      return;
    }
    if (form.password !== form.password_confirm) {
      setErrors({ password_confirm: 'Passwords do not match.' });
      return;
    }
    setLoading(true);
    setErrors({});
    try {
      const res = await register(form);
      setAuth(res.data.user, res.data.token);
      navigate('/');
    } catch (err: any) {
      const data = err.response?.data;
      if (typeof data === 'object' && data) {
        const fieldErrors: Record<string, string> = {};
        for (const [key, val] of Object.entries(data)) {
          fieldErrors[key] = Array.isArray(val) ? (val as string[]).join(' ') : String(val);
        }
        setErrors(fieldErrors);
      } else {
        setErrors({ general: 'Registration failed. Please try again.' });
      }
    } finally {
      setLoading(false);
    }
  };

  const update = (field: string) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm({ ...form, [field]: e.target.value });

  return (
    <Box sx={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'linear-gradient(135deg, #0d47a1 0%, #1565c0 50%, #1976d2 100%)',
      p: 2,
    }}>
      <Card sx={{ maxWidth: 500, width: '100%', borderRadius: 3 }}>
        <CardContent sx={{ p: { xs: 3, sm: 4 } }}>
          <Box sx={{ textAlign: 'center', mb: 3 }}>
            <HospitalIcon sx={{ fontSize: 48, color: 'primary.main', mb: 1 }} />
            <Typography variant="h5" fontWeight={700}>Create Account</Typography>
            <Typography variant="body2" color="text.secondary">
              Join MS Risk Lab
            </Typography>
          </Box>

          {errors.general && <Alert severity="error" sx={{ mb: 2 }}>{errors.general}</Alert>}

          <form onSubmit={handleSubmit}>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <TextField
                  label="First Name"
                  value={form.first_name}
                  onChange={update('first_name')}
                  fullWidth
                  error={!!errors.first_name}
                  helperText={errors.first_name}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  label="Last Name"
                  value={form.last_name}
                  onChange={update('last_name')}
                  fullWidth
                  error={!!errors.last_name}
                  helperText={errors.last_name}
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  label="Username *"
                  value={form.username}
                  onChange={update('username')}
                  fullWidth
                  error={!!errors.username}
                  helperText={errors.username}
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  label="Email"
                  type="email"
                  value={form.email}
                  onChange={update('email')}
                  fullWidth
                  error={!!errors.email}
                  helperText={errors.email}
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  label="Password *"
                  type={showPassword ? 'text' : 'password'}
                  value={form.password}
                  onChange={update('password')}
                  fullWidth
                  error={!!errors.password}
                  helperText={errors.password}
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
              </Grid>
              <Grid item xs={12}>
                <TextField
                  label="Confirm Password *"
                  type={showPassword ? 'text' : 'password'}
                  value={form.password_confirm}
                  onChange={update('password_confirm')}
                  fullWidth
                  error={!!errors.password_confirm}
                  helperText={errors.password_confirm}
                />
              </Grid>
            </Grid>

            <Button
              type="submit"
              variant="contained"
              fullWidth
              size="large"
              disabled={loading}
              startIcon={<RegisterIcon />}
              sx={{ mt: 3, mb: 2, py: 1.5, borderRadius: 2 }}
            >
              {loading ? 'Creating account...' : 'Create Account'}
            </Button>
          </form>

          <Typography variant="body2" align="center" color="text.secondary">
            Already have an account?{' '}
            <Link component={RouterLink} to="/login" underline="hover">
              Sign in
            </Link>
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
}
