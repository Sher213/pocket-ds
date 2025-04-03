import { useState } from 'react';
import {
  Box,
  Paper,
  Grid,
  Typography,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
} from 'recharts';

const mockData = [
  { x: 10, y: 30, z: 200 },
  { x: 20, y: 50, z: 150 },
  { x: 30, y: 40, z: 180 },
  { x: 40, y: 70, z: 220 },
  { x: 50, y: 60, z: 190 },
  { x: 60, y: 90, z: 250 },
  { x: 70, y: 80, z: 210 },
];

const chartTypes = ['line', 'scatter', 'bar', 'histogram'] as const;
type ChartType = typeof chartTypes[number];

export default function Analysis() {
  const [selectedDataset, setSelectedDataset] = useState('dataset1');
  const [chartType, setChartType] = useState<ChartType>('line');
  const [xAxis, setXAxis] = useState('x');
  const [yAxis, setYAxis] = useState('y');

  return (
    <Box sx={{ flexGrow: 1, p: 2 }}>
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Data Analysis
            </Typography>
            <Grid container spacing={2} sx={{ mb: 3 }}>
              <Grid item xs={12} sm={3}>
                <FormControl fullWidth>
                  <InputLabel>Dataset</InputLabel>
                  <Select
                    value={selectedDataset}
                    label="Dataset"
                    onChange={(e) => setSelectedDataset(e.target.value)}
                  >
                    <MenuItem value="dataset1">Dataset 1</MenuItem>
                    <MenuItem value="dataset2">Dataset 2</MenuItem>
                    <MenuItem value="dataset3">Dataset 3</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={3}>
                <FormControl fullWidth>
                  <InputLabel>Chart Type</InputLabel>
                  <Select
                    value={chartType}
                    label="Chart Type"
                    onChange={(e) => setChartType(e.target.value as ChartType)}
                  >
                    {chartTypes.map((type) => (
                      <MenuItem key={type} value={type}>
                        {type.charAt(0).toUpperCase() + type.slice(1)}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={3}>
                <FormControl fullWidth>
                  <InputLabel>X Axis</InputLabel>
                  <Select
                    value={xAxis}
                    label="X Axis"
                    onChange={(e) => setXAxis(e.target.value)}
                  >
                    <MenuItem value="x">X Value</MenuItem>
                    <MenuItem value="y">Y Value</MenuItem>
                    <MenuItem value="z">Z Value</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={3}>
                <FormControl fullWidth>
                  <InputLabel>Y Axis</InputLabel>
                  <Select
                    value={yAxis}
                    label="Y Axis"
                    onChange={(e) => setYAxis(e.target.value)}
                  >
                    <MenuItem value="x">X Value</MenuItem>
                    <MenuItem value="y">Y Value</MenuItem>
                    <MenuItem value="z">Z Value</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Paper sx={{ p: 2, height: 400 }}>
            <ResponsiveContainer width="100%" height="100%">
              {chartType === 'line' ? (
                <LineChart data={mockData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey={xAxis} />
                  <YAxis dataKey={yAxis} />
                  <Tooltip />
                  <Line type="monotone" dataKey={yAxis} stroke="#1976d2" />
                </LineChart>
              ) : chartType === 'scatter' ? (
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey={xAxis} />
                  <YAxis dataKey={yAxis} />
                  <Tooltip />
                  <Scatter data={mockData} fill="#1976d2" />
                </ScatterChart>
              ) : (
                <Typography>Chart type not implemented yet</Typography>
              )}
            </ResponsiveContainer>
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Statistical Summary
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={3}>
                <Typography variant="body2" color="text.secondary">
                  Mean:
                </Typography>
                <Typography variant="body1">45.2</Typography>
              </Grid>
              <Grid item xs={12} sm={3}>
                <Typography variant="body2" color="text.secondary">
                  Median:
                </Typography>
                <Typography variant="body1">42.8</Typography>
              </Grid>
              <Grid item xs={12} sm={3}>
                <Typography variant="body2" color="text.secondary">
                  Standard Deviation:
                </Typography>
                <Typography variant="body1">12.4</Typography>
              </Grid>
              <Grid item xs={12} sm={3}>
                <Typography variant="body2" color="text.secondary">
                  Sample Size:
                </Typography>
                <Typography variant="body1">1,234</Typography>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
} 