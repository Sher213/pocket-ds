import { Grid, Paper, Typography, Box } from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

const mockData = [
  { name: 'Dataset 1', size: 2000 },
  { name: 'Dataset 2', size: 4000 },
  { name: 'Dataset 3', size: 1500 },
  { name: 'Dataset 4', size: 3000 },
];

const DashboardCard = ({ title, value, description }: { title: string; value: string; description: string }) => (
  <Paper
    sx={{
      p: 2,
      display: 'flex',
      flexDirection: 'column',
      height: 140,
    }}
  >
    <Typography color="text.secondary" gutterBottom>
      {title}
    </Typography>
    <Typography component="h2" variant="h3" color="primary">
      {value}
    </Typography>
    <Typography color="text.secondary" sx={{ flex: 1 }}>
      {description}
    </Typography>
  </Paper>
);

export default function Dashboard() {
  return (
    <Box sx={{ flexGrow: 1 }}>
      <Grid container spacing={3}>
        <Grid item xs={12} md={3}>
          <DashboardCard
            title="Total Datasets"
            value="4"
            description="Active datasets in your workspace"
          />
        </Grid>
        <Grid item xs={12} md={3}>
          <DashboardCard
            title="Models"
            value="2"
            description="Trained machine learning models"
          />
        </Grid>
        <Grid item xs={12} md={3}>
          <DashboardCard
            title="Reports"
            value="5"
            description="Generated analysis reports"
          />
        </Grid>
        <Grid item xs={12} md={3}>
          <DashboardCard
            title="Storage"
            value="1.2GB"
            description="Total data storage used"
          />
        </Grid>
        <Grid item xs={12}>
          <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column', height: 400 }}>
            <Typography variant="h6" gutterBottom>
              Dataset Overview
            </Typography>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={mockData}
                margin={{
                  top: 20,
                  right: 30,
                  left: 20,
                  bottom: 5,
                }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="size" fill="#1976d2" />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
} 