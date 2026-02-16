import React, { useEffect, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Box, Drawer, AppBar, Toolbar, Typography, List, ListItem,
  ListItemButton, ListItemIcon, ListItemText, IconButton, Badge,
  Divider, Chip, Tooltip, Avatar, useTheme, useMediaQuery,
  Menu, MenuItem,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  People as PeopleIcon,
  Assessment as AssessmentIcon,
  PlayArrow as WorkflowIcon,
  Balance as FairnessIcon,
  Tune as WhatIfIcon,
  Policy as PolicyIcon,
  Security as GovernanceIcon,
  SmartToy as AgentsIcon,
  History as AuditIcon,
  Notifications as NotificationsIcon,
  Menu as MenuIcon,
  DarkMode as DarkModeIcon,
  LightMode as LightModeIcon,
  LocalHospital as HospitalIcon,
  Person as PersonIcon,
  Logout as LogoutIcon,
} from '@mui/icons-material';
import { useAppStore } from '../store';
import { getDashboard, getUnreadCount, logout } from '../services/api';

const DRAWER_WIDTH = 260;

const menuItems = [
  { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
  { text: 'Patients', icon: <PeopleIcon />, path: '/patients' },
  { text: 'Risk Assessments', icon: <AssessmentIcon />, path: '/assessments' },
  { text: 'Workflow Runs', icon: <WorkflowIcon />, path: '/workflows' },
  { divider: true },
  { text: 'Fairness Analysis', icon: <FairnessIcon />, path: '/fairness' },
  { text: 'What-If Analysis', icon: <WhatIfIcon />, path: '/what-if' },
  { text: 'Policy Config', icon: <PolicyIcon />, path: '/policies' },
  { divider: true },
  { text: 'Governance Rules', icon: <GovernanceIcon />, path: '/governance' },
  { text: 'AI Agents', icon: <AgentsIcon />, path: '/agents' },
  { text: 'Audit Trail', icon: <AuditIcon />, path: '/audit' },
  { text: 'Notifications', icon: <NotificationsIcon />, path: '/notifications' },
  { divider: true },
  { text: 'Profile', icon: <PersonIcon />, path: '/profile' },
];

interface LayoutProps {
  children: React.ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const { sidebarOpen, toggleSidebar, setSidebarOpen, darkMode, toggleDarkMode, unreadCount, setUnreadCount, setDashboard, user, clearAuth } = useAppStore();
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  const userInitials = user
    ? (user.first_name && user.last_name
        ? `${user.first_name[0]}${user.last_name[0]}`
        : user.username.substring(0, 2)
      ).toUpperCase()
    : '?';

  const displayName = user
    ? (user.first_name && user.last_name
        ? `${user.first_name} ${user.last_name}`
        : user.username)
    : '';

  const handleLogout = async () => {
    setAnchorEl(null);
    try { await logout(); } catch {}
    clearAuth();
    navigate('/login');
  };

  useEffect(() => {
    getDashboard().then((r) => setDashboard(r.data)).catch(() => {});
    getUnreadCount().then((r) => setUnreadCount(r.data.unread_count)).catch(() => {});
    const interval = setInterval(() => {
      getUnreadCount().then((r) => setUnreadCount(r.data.unread_count)).catch(() => {});
    }, 30000);
    return () => clearInterval(interval);
  }, [setDashboard, setUnreadCount]);

  const handleNavigation = (path: string) => {
    navigate(path);
    if (isMobile) {
      setSidebarOpen(false);
    }
  };

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <AppBar
        position="fixed"
        sx={{
          zIndex: theme.zIndex.drawer + 1,
          background: darkMode
            ? 'linear-gradient(135deg, #0d47a1 0%, #1565c0 100%)'
            : 'linear-gradient(135deg, #1565c0 0%, #1976d2 100%)',
        }}
      >
        <Toolbar sx={{ minHeight: { xs: 56, sm: 64 }, px: { xs: 1, sm: 2 } }}>
          <IconButton color="inherit" edge="start" onClick={toggleSidebar} sx={{ mr: { xs: 0.5, sm: 2 } }}>
            <MenuIcon />
          </IconButton>
          <HospitalIcon sx={{ mr: 1, display: { xs: 'none', sm: 'block' } }} />
          <Typography variant="h6" noWrap sx={{ flexGrow: 1, fontWeight: 700, fontSize: { xs: '1rem', sm: '1.25rem' } }}>
            MS Risk Lab
            <Chip
              label="AI Healthcare Assistant"
              size="small"
              sx={{ ml: 1.5, bgcolor: 'rgba(255,255,255,0.15)', color: 'white', fontWeight: 500, display: { xs: 'none', md: 'inline-flex' } }}
            />
          </Typography>
          <Tooltip title={darkMode ? 'Light Mode' : 'Dark Mode'}>
            <IconButton color="inherit" onClick={toggleDarkMode}>
              {darkMode ? <LightModeIcon /> : <DarkModeIcon />}
            </IconButton>
          </Tooltip>
          <Tooltip title="Notifications">
            <IconButton color="inherit" onClick={() => handleNavigation('/notifications')}>
              <Badge badgeContent={unreadCount} color="error">
                <NotificationsIcon />
              </Badge>
            </IconButton>
          </Tooltip>
          <Tooltip title={displayName || 'Account'}>
            <IconButton onClick={(e) => setAnchorEl(e.currentTarget)} sx={{ ml: 0.5, p: 0 }}>
              <Avatar sx={{ bgcolor: 'rgba(255,255,255,0.2)', width: { xs: 30, sm: 36 }, height: { xs: 30, sm: 36 }, fontSize: { xs: '0.75rem', sm: '0.875rem' } }}>
                {userInitials}
              </Avatar>
            </IconButton>
          </Tooltip>
          <Menu
            anchorEl={anchorEl}
            open={Boolean(anchorEl)}
            onClose={() => setAnchorEl(null)}
            anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
            transformOrigin={{ vertical: 'top', horizontal: 'right' }}
          >
            <MenuItem disabled sx={{ opacity: '1 !important' }}>
              <Typography variant="body2" fontWeight={600}>{displayName}</Typography>
            </MenuItem>
            <Divider />
            <MenuItem onClick={() => { setAnchorEl(null); handleNavigation('/profile'); }}>
              <ListItemIcon><PersonIcon fontSize="small" /></ListItemIcon>
              Profile
            </MenuItem>
            <MenuItem onClick={handleLogout}>
              <ListItemIcon><LogoutIcon fontSize="small" /></ListItemIcon>
              Logout
            </MenuItem>
          </Menu>
        </Toolbar>
      </AppBar>

      <Drawer
        variant={isMobile ? 'temporary' : 'persistent'}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        ModalProps={{ keepMounted: true }}
        sx={{
          width: sidebarOpen && !isMobile ? DRAWER_WIDTH : 0,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: DRAWER_WIDTH,
            boxSizing: 'border-box',
            borderRight: 'none',
            bgcolor: darkMode ? '#0d1b2a' : '#fafbfc',
          },
        }}
      >
        <Toolbar sx={{ minHeight: { xs: 56, sm: 64 } }} />
        <Box sx={{ overflow: 'auto', mt: 1 }}>
          <List>
            {menuItems.map((item, index) =>
              'divider' in item ? (
                <Divider key={index} sx={{ my: 1 }} />
              ) : (
                <ListItem key={item.text} disablePadding sx={{ px: 1 }}>
                  <ListItemButton
                    selected={location.pathname === item.path}
                    onClick={() => handleNavigation(item.path!)}
                    sx={{
                      borderRadius: 2,
                      mb: 0.5,
                      '&.Mui-selected': {
                        bgcolor: darkMode ? 'rgba(25,118,210,0.2)' : 'rgba(25,118,210,0.08)',
                        '&:hover': {
                          bgcolor: darkMode ? 'rgba(25,118,210,0.3)' : 'rgba(25,118,210,0.12)',
                        },
                      },
                    }}
                  >
                    <ListItemIcon sx={{ minWidth: 40, color: location.pathname === item.path ? 'primary.main' : 'text.secondary' }}>
                      {item.text === 'Notifications' ? (
                        <Badge badgeContent={unreadCount} color="error" variant="dot">
                          {item.icon}
                        </Badge>
                      ) : (
                        item.icon
                      )}
                    </ListItemIcon>
                    <ListItemText
                      primary={item.text}
                      primaryTypographyProps={{
                        fontSize: '0.875rem',
                        fontWeight: location.pathname === item.path ? 600 : 400,
                      }}
                    />
                  </ListItemButton>
                </ListItem>
              )
            )}
          </List>
        </Box>
      </Drawer>

      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: { xs: 1.5, sm: 2, md: 3 },
          mt: { xs: 7, sm: 8 },
          ml: !isMobile && sidebarOpen ? 0 : isMobile ? 0 : `-${DRAWER_WIDTH}px`,
          transition: 'margin 0.3s',
          minHeight: 'calc(100vh - 64px)',
          overflow: 'hidden',
          width: '100%',
          maxWidth: '100vw',
          boxSizing: 'border-box',
        }}
      >
        {children}
      </Box>
    </Box>
  );
}
