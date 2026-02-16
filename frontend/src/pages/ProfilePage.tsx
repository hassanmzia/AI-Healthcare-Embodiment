import React, { useState } from 'react';
import {
  Box, Typography, Card, CardContent, Grid, TextField, Button,
  Alert, Divider, InputAdornment, IconButton,
} from '@mui/material';
import {
  Save as SaveIcon, Lock as LockIcon,
  Visibility, VisibilityOff,
} from '@mui/icons-material';
import { updateProfile, changePassword } from '../services/api';
import { useAppStore } from '../store';

export default function ProfilePage() {
  const { user, setAuth } = useAppStore();
  const [profile, setProfile] = useState({
    email: user?.email || '',
    first_name: user?.first_name || '',
    last_name: user?.last_name || '',
  });
  const [profileMsg, setProfileMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [profileLoading, setProfileLoading] = useState(false);

  const [passwords, setPasswords] = useState({
    old_password: '', new_password: '', new_password_confirm: '',
  });
  const [pwMsg, setPwMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [pwLoading, setPwLoading] = useState(false);
  const [showPasswords, setShowPasswords] = useState(false);

  const handleProfileSave = async (e: React.FormEvent) => {
    e.preventDefault();
    setProfileLoading(true);
    setProfileMsg(null);
    try {
      const res = await updateProfile(profile);
      const token = localStorage.getItem('auth_token') || '';
      setAuth(res.data, token);
      setProfileMsg({ type: 'success', text: 'Profile updated successfully.' });
    } catch (err: any) {
      const msg = err.response?.data ? Object.values(err.response.data).flat().join(' ') : 'Failed to update profile.';
      setProfileMsg({ type: 'error', text: msg });
    } finally {
      setProfileLoading(false);
    }
  };

  const handlePasswordChange = async (e: React.FormEvent) => {
    e.preventDefault();
    if (passwords.new_password !== passwords.new_password_confirm) {
      setPwMsg({ type: 'error', text: 'New passwords do not match.' });
      return;
    }
    setPwLoading(true);
    setPwMsg(null);
    try {
      const res = await changePassword(passwords);
      // Update token after password change
      const token = res.data.token;
      if (user) setAuth(user, token);
      setPasswords({ old_password: '', new_password: '', new_password_confirm: '' });
      setPwMsg({ type: 'success', text: 'Password changed successfully.' });
    } catch (err: any) {
      const data = err.response?.data;
      let msg = 'Failed to change password.';
      if (typeof data === 'object' && data) {
        msg = Object.values(data).flat().join(' ');
      }
      setPwMsg({ type: 'error', text: msg });
    } finally {
      setPwLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ fontSize: { xs: '1.5rem', sm: '2.125rem' } }}>
        Profile Settings
      </Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        Manage your account information and security
      </Typography>

      <Grid container spacing={3} sx={{ mt: 1 }}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Personal Information</Typography>
              {profileMsg && <Alert severity={profileMsg.type} sx={{ mb: 2 }}>{profileMsg.text}</Alert>}
              <form onSubmit={handleProfileSave}>
                <TextField
                  label="Username"
                  value={user?.username || ''}
                  fullWidth
                  disabled
                  sx={{ mb: 2 }}
                  helperText="Username cannot be changed"
                />
                <Grid container spacing={2} sx={{ mb: 2 }}>
                  <Grid item xs={12} sm={6}>
                    <TextField
                      label="First Name"
                      value={profile.first_name}
                      onChange={(e) => setProfile({ ...profile, first_name: e.target.value })}
                      fullWidth
                    />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <TextField
                      label="Last Name"
                      value={profile.last_name}
                      onChange={(e) => setProfile({ ...profile, last_name: e.target.value })}
                      fullWidth
                    />
                  </Grid>
                </Grid>
                <TextField
                  label="Email"
                  type="email"
                  value={profile.email}
                  onChange={(e) => setProfile({ ...profile, email: e.target.value })}
                  fullWidth
                  sx={{ mb: 2 }}
                />
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 2 }}>
                  Member since {user?.date_joined ? new Date(user.date_joined).toLocaleDateString() : 'N/A'}
                  {user?.is_staff && ' | Staff account'}
                </Typography>
                <Button
                  type="submit"
                  variant="contained"
                  startIcon={<SaveIcon />}
                  disabled={profileLoading}
                >
                  {profileLoading ? 'Saving...' : 'Save Changes'}
                </Button>
              </form>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Change Password</Typography>
              {pwMsg && <Alert severity={pwMsg.type} sx={{ mb: 2 }}>{pwMsg.text}</Alert>}
              <form onSubmit={handlePasswordChange}>
                <TextField
                  label="Current Password"
                  type={showPasswords ? 'text' : 'password'}
                  value={passwords.old_password}
                  onChange={(e) => setPasswords({ ...passwords, old_password: e.target.value })}
                  fullWidth
                  sx={{ mb: 2 }}
                />
                <TextField
                  label="New Password"
                  type={showPasswords ? 'text' : 'password'}
                  value={passwords.new_password}
                  onChange={(e) => setPasswords({ ...passwords, new_password: e.target.value })}
                  fullWidth
                  sx={{ mb: 2 }}
                  InputProps={{
                    endAdornment: (
                      <InputAdornment position="end">
                        <IconButton onClick={() => setShowPasswords(!showPasswords)} edge="end">
                          {showPasswords ? <VisibilityOff /> : <Visibility />}
                        </IconButton>
                      </InputAdornment>
                    ),
                  }}
                />
                <TextField
                  label="Confirm New Password"
                  type={showPasswords ? 'text' : 'password'}
                  value={passwords.new_password_confirm}
                  onChange={(e) => setPasswords({ ...passwords, new_password_confirm: e.target.value })}
                  fullWidth
                  sx={{ mb: 3 }}
                />
                <Button
                  type="submit"
                  variant="contained"
                  color="warning"
                  startIcon={<LockIcon />}
                  disabled={pwLoading || !passwords.old_password || !passwords.new_password}
                >
                  {pwLoading ? 'Changing...' : 'Change Password'}
                </Button>
              </form>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}
