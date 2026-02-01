import React, { useEffect, useState } from 'react';
import {
  Box, Typography, Card, CircularProgress, TextField, MenuItem,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  TablePagination, Chip, Dialog, DialogTitle, DialogContent,
  DialogActions, Button, IconButton, Divider, Grid,
} from '@mui/material';
import { Close as CloseIcon } from '@mui/icons-material';
import { getAuditLogs } from '../services/api';
import { formatDate } from '../utils/helpers';
import type { AuditLog } from '../types';

export default function AuditPage() {
  const [logs, setLogs] = useState<AuditLog[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(25);
  const [loading, setLoading] = useState(true);
  const [typeFilter, setTypeFilter] = useState('');
  const [selectedLog, setSelectedLog] = useState<AuditLog | null>(null);

  useEffect(() => {
    setLoading(true);
    const params: Record<string, any> = { page: page + 1, page_size: rowsPerPage };
    if (typeFilter) params.action_type = typeFilter;
    getAuditLogs(params)
      .then((r) => { setLogs(r.data.results); setTotal(r.data.count); })
      .finally(() => setLoading(false));
  }, [page, rowsPerPage, typeFilter]);

  const typeColors: Record<string, any> = {
    AGENT_RUN: 'primary', DECISION: 'info', POLICY_CHANGE: 'warning',
    MANUAL_REVIEW: 'success', OVERRIDE: 'error', ALERT: 'error', EXPORT: 'default',
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>Audit Trail</Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        Complete audit log of all system actions, decisions, and reviews
      </Typography>

      <Card sx={{ mb: 2 }}>
        <Box sx={{ p: 2 }}>
          <TextField select label="Action Type" value={typeFilter} size="small" sx={{ minWidth: 200 }}
            onChange={(e) => { setTypeFilter(e.target.value); setPage(0); }}>
            <MenuItem value="">All Types</MenuItem>
            <MenuItem value="AGENT_RUN">Agent Run</MenuItem>
            <MenuItem value="DECISION">Decision</MenuItem>
            <MenuItem value="POLICY_CHANGE">Policy Change</MenuItem>
            <MenuItem value="MANUAL_REVIEW">Manual Review</MenuItem>
            <MenuItem value="OVERRIDE">Override</MenuItem>
            <MenuItem value="ALERT">Alert</MenuItem>
          </TextField>
        </Box>
      </Card>

      <Card>
        <TableContainer>
          {loading ? <Box sx={{ p: 4, textAlign: 'center' }}><CircularProgress /></Box> : (
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Timestamp</TableCell>
                  <TableCell>Action Type</TableCell>
                  <TableCell>Actor</TableCell>
                  <TableCell>Target</TableCell>
                  <TableCell>Details</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {logs.map((log) => (
                  <TableRow key={log.id} hover sx={{ cursor: 'pointer' }} onClick={() => setSelectedLog(log)}>
                    <TableCell sx={{ fontSize: '0.8rem', whiteSpace: 'nowrap' }}>{formatDate(log.created_at)}</TableCell>
                    <TableCell><Chip label={log.action_type} size="small" color={typeColors[log.action_type] || 'default'} /></TableCell>
                    <TableCell>{log.actor}</TableCell>
                    <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
                      {log.target_type}{log.target_id ? `:${log.target_id.slice(0, 8)}` : ''}
                    </TableCell>
                    <TableCell sx={{ maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: '0.8rem' }}>
                      {JSON.stringify(log.details)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </TableContainer>
        <TablePagination
          rowsPerPageOptions={[10, 25, 50]} component="div" count={total}
          rowsPerPage={rowsPerPage} page={page}
          onPageChange={(_, p) => setPage(p)}
          onRowsPerPageChange={(e) => { setRowsPerPage(parseInt(e.target.value)); setPage(0); }}
        />
      </Card>

      {/* Audit Log Detail Dialog */}
      <Dialog open={!!selectedLog} onClose={() => setSelectedLog(null)} maxWidth="sm" fullWidth>
        {selectedLog && (
          <>
            <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Box>
                <Typography variant="h6">Audit Log Entry</Typography>
                <Chip label={selectedLog.action_type} size="small" color={typeColors[selectedLog.action_type] || 'default'} sx={{ mt: 0.5 }} />
              </Box>
              <IconButton onClick={() => setSelectedLog(null)} size="small"><CloseIcon /></IconButton>
            </DialogTitle>
            <DialogContent dividers>
              <Grid container spacing={2} sx={{ mb: 2 }}>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">Timestamp</Typography>
                  <Typography variant="body2">{formatDate(selectedLog.created_at)}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">Actor</Typography>
                  <Typography variant="body2">{selectedLog.actor || 'N/A'}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">Target Type</Typography>
                  <Typography variant="body2">{selectedLog.target_type || 'N/A'}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">Target ID</Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.85rem' }}>
                    {selectedLog.target_id || 'N/A'}
                  </Typography>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="caption" color="text.secondary">Log ID</Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.85rem' }}>
                    {selectedLog.id}
                  </Typography>
                </Grid>
              </Grid>
              <Divider sx={{ my: 1 }} />
              <Typography variant="subtitle2" sx={{ mt: 1, mb: 1 }}>Details</Typography>
              <Box sx={{
                bgcolor: 'grey.50', borderRadius: 1, p: 2, fontFamily: 'monospace',
                fontSize: '0.85rem', whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                border: '1px solid', borderColor: 'grey.200', maxHeight: 400, overflow: 'auto',
              }}>
                {selectedLog.details
                  ? JSON.stringify(selectedLog.details, null, 2)
                  : 'No details available'}
              </Box>
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setSelectedLog(null)}>Close</Button>
            </DialogActions>
          </>
        )}
      </Dialog>
    </Box>
  );
}
