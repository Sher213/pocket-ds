import { useState } from 'react';
import {
  Box,
  Paper,
  Grid,
  Typography,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
} from '@mui/material';
import {
  Description as ReportIcon,
  Download as DownloadIcon,
  Share as ShareIcon,
  Delete as DeleteIcon,
} from '@mui/icons-material';

interface Report {
  id: string;
  title: string;
  type: string;
  date: string;
  status: 'completed' | 'generating';
}

const mockReports: Report[] = [
  {
    id: '1',
    title: 'Housing Price Analysis Report',
    type: 'Regression Analysis',
    date: '2024-03-15',
    status: 'completed',
  },
  {
    id: '2',
    title: 'Customer Segmentation Results',
    type: 'Clustering Analysis',
    date: '2024-03-14',
    status: 'completed',
  },
  {
    id: '3',
    title: 'Stock Market Prediction',
    type: 'Time Series Analysis',
    date: '2024-03-13',
    status: 'generating',
  },
];

export default function Reports() {
  const [reports, setReports] = useState<Report[]>(mockReports);
  const [openNewReport, setOpenNewReport] = useState(false);
  const [newReportTitle, setNewReportTitle] = useState('');
  const [newReportType, setNewReportType] = useState('');

  const handleCreateReport = () => {
    const newReport: Report = {
      id: String(reports.length + 1),
      title: newReportTitle,
      type: newReportType,
      date: new Date().toISOString().split('T')[0],
      status: 'generating',
    };
    setReports([newReport, ...reports]);
    setOpenNewReport(false);
    setNewReportTitle('');
    setNewReportType('');
  };

  const handleDeleteReport = (id: string) => {
    setReports(reports.filter((report) => report.id !== id));
  };

  return (
    <Box sx={{ flexGrow: 1, p: 2 }}>
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
            <Typography variant="h5">Analysis Reports</Typography>
            <Button
              variant="contained"
              startIcon={<ReportIcon />}
              onClick={() => setOpenNewReport(true)}
            >
              Generate New Report
            </Button>
          </Box>
        </Grid>

        <Grid item xs={12}>
          <Paper>
            <List>
              {reports.map((report) => (
                <ListItem key={report.id} divider>
                  <ListItemText
                    primary={report.title}
                    secondary={
                      <Box sx={{ mt: 1 }}>
                        <Chip
                          label={report.type}
                          size="small"
                          sx={{ mr: 1 }}
                        />
                        <Chip
                          label={report.date}
                          size="small"
                          sx={{ mr: 1 }}
                        />
                        <Chip
                          label={report.status}
                          color={report.status === 'completed' ? 'success' : 'warning'}
                          size="small"
                        />
                      </Box>
                    }
                  />
                  <ListItemSecondaryAction>
                    <IconButton
                      edge="end"
                      aria-label="download"
                      sx={{ mr: 1 }}
                      disabled={report.status === 'generating'}
                    >
                      <DownloadIcon />
                    </IconButton>
                    <IconButton
                      edge="end"
                      aria-label="share"
                      sx={{ mr: 1 }}
                      disabled={report.status === 'generating'}
                    >
                      <ShareIcon />
                    </IconButton>
                    <IconButton
                      edge="end"
                      aria-label="delete"
                      onClick={() => handleDeleteReport(report.id)}
                    >
                      <DeleteIcon />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>
      </Grid>

      <Dialog open={openNewReport} onClose={() => setOpenNewReport(false)}>
        <DialogTitle>Generate New Report</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 2 }}>
            <TextField
              autoFocus
              margin="dense"
              label="Report Title"
              fullWidth
              value={newReportTitle}
              onChange={(e) => setNewReportTitle(e.target.value)}
              sx={{ mb: 2 }}
            />
            <FormControl fullWidth>
              <InputLabel>Report Type</InputLabel>
              <Select
                value={newReportType}
                label="Report Type"
                onChange={(e) => setNewReportType(e.target.value)}
              >
                <MenuItem value="Regression Analysis">Regression Analysis</MenuItem>
                <MenuItem value="Classification Report">Classification Report</MenuItem>
                <MenuItem value="Clustering Analysis">Clustering Analysis</MenuItem>
                <MenuItem value="Time Series Analysis">Time Series Analysis</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenNewReport(false)}>Cancel</Button>
          <Button
            onClick={handleCreateReport}
            variant="contained"
            disabled={!newReportTitle || !newReportType}
          >
            Generate
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
} 