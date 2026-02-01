import React, { useEffect, useState } from 'react';
import {
  Box, Typography, Card, CardContent, Button, CircularProgress,
  List, ListItem, ListItemIcon, ListItemText, Chip, Divider, IconButton,
} from '@mui/material';
import {
  Info as InfoIcon, Warning as WarningIcon, Error as ErrorIcon,
  CheckCircle as SuccessIcon, MarkEmailRead as ReadIcon, DoneAll as DoneAllIcon,
} from '@mui/icons-material';
import { useAppStore } from '../store';
import { getNotifications, markNotificationRead, markAllNotificationsRead, getUnreadCount } from '../services/api';
import { formatDate, getSeverityColor } from '../utils/helpers';
import type { Notification } from '../types';

export default function NotificationsPage() {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [loading, setLoading] = useState(true);
  const { setUnreadCount } = useAppStore();

  const load = () => {
    setLoading(true);
    getNotifications().then((r) => setNotifications(r.data.results || [])).finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  const handleMarkRead = async (id: string) => {
    await markNotificationRead(id);
    load();
    getUnreadCount().then((r) => setUnreadCount(r.data.unread_count));
  };

  const handleMarkAllRead = async () => {
    await markAllNotificationsRead();
    load();
    setUnreadCount(0);
  };

  const severityIcon = (s: string) => {
    switch (s) {
      case 'critical': return <ErrorIcon color="error" />;
      case 'warning': return <WarningIcon color="warning" />;
      case 'success': return <SuccessIcon color="success" />;
      default: return <InfoIcon color="info" />;
    }
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4">Notifications</Typography>
          <Typography variant="body2" color="text.secondary">System alerts and clinician notifications</Typography>
        </Box>
        <Button variant="outlined" startIcon={<DoneAllIcon />} onClick={handleMarkAllRead}>Mark All Read</Button>
      </Box>

      <Card>
        {loading ? <Box sx={{ p: 4, textAlign: 'center' }}><CircularProgress /></Box> : (
          <List>
            {notifications.length === 0 && (
              <ListItem><ListItemText primary="No notifications" secondary="You're all caught up!" /></ListItem>
            )}
            {notifications.map((n, i) => (
              <React.Fragment key={n.id}>
                {i > 0 && <Divider />}
                <ListItem
                  sx={{ bgcolor: n.is_read ? 'transparent' : 'action.hover', py: 2 }}
                  secondaryAction={
                    !n.is_read && (
                      <IconButton onClick={() => handleMarkRead(n.id)}><ReadIcon /></IconButton>
                    )
                  }
                >
                  <ListItemIcon>{severityIcon(n.severity)}</ListItemIcon>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                        <Typography variant="subtitle2" sx={{ fontWeight: n.is_read ? 400 : 700 }}>{n.title}</Typography>
                        <Chip label={n.category} size="small" variant="outlined" />
                        {!n.is_read && <Chip label="New" size="small" color="primary" />}
                        {n.related_patient_id && (
                          <Chip label={`Patient: ${n.related_patient_id}`} size="small" color="info" variant="outlined" />
                        )}
                      </Box>
                    }
                    secondary={
                      <Box>
                        <Typography variant="body2" sx={{ mb: 0.5 }}>{n.message}</Typography>
                        {n.metadata?.patient_ids && (n.metadata.patient_ids as string[]).length > 0 && (
                          <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mb: 0.5 }}>
                            {(n.metadata.patient_ids as string[]).slice(0, 8).map((pid: string) => (
                              <Chip key={pid} label={pid} size="small" sx={{ fontFamily: 'monospace', fontSize: '0.7rem', height: 20 }} />
                            ))}
                            {(n.metadata.patient_ids as string[]).length > 8 && (
                              <Chip label={`+${(n.metadata.patient_ids as string[]).length - 8} more`} size="small" sx={{ fontSize: '0.7rem', height: 20 }} />
                            )}
                          </Box>
                        )}
                        <Typography variant="caption" color="text.secondary">{formatDate(n.created_at)}</Typography>
                      </Box>
                    }
                  />
                </ListItem>
              </React.Fragment>
            ))}
          </List>
        )}
      </Card>
    </Box>
  );
}
