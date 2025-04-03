import { useState, useEffect } from 'react';
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
  Stepper,
  Step,
  StepLabel,
  CircularProgress,
  Slider,
  Chip,
  Alert,
  Snackbar,
  TextField,
  FormHelperText,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { dataService, mlService, type TrainingRequest } from '../services/api';

interface Dataset {
  id: string;
  name: string;
  columns: string[];
  shape: [number, number];
}

export default function MachineLearning() {
  const [activeStep, setActiveStep] = useState(0);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [selectedModel, setSelectedModel] = useState('');
  const [targetColumn, setTargetColumn] = useState('');
  const [featureColumns, setFeatureColumns] = useState<string[]>([]);
  const [epochs, setEpochs] = useState(10);
  const [learningRate, setLearningRate] = useState(0.01);
  const [testSize, setTestSize] = useState(0.2);
  const [isTraining, setIsTraining] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [trainingMetrics, setTrainingMetrics] = useState<any>(null);

  const steps = ['Select Data', 'Choose Model', 'Configure Parameters', 'Train Model'];

  useEffect(() => {
    loadDatasets();
  }, []);

  const loadDatasets = async () => {
    try {
      const response = await dataService.listDatasets();
      setDatasets(response.datasets);
    } catch (err) {
      setError('Failed to load datasets');
    }
  };

  const handleNext = () => {
    setActiveStep((prevStep) => prevStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };

  const handleTrain = async () => {
    if (!selectedDataset) return;

    setIsTraining(true);
    setError(null);
    setSuccess(null);

    try {
      const trainingRequest: TrainingRequest = {
        dataset_id: selectedDataset.id,
        model_type: selectedModel as TrainingRequest['model_type'],
        target_column: targetColumn,
        feature_columns: featureColumns,
        test_size: testSize,
        hyperparameters: {
          n_epochs: epochs,
          learning_rate: learningRate,
        },
      };

      const response = await mlService.trainModel(trainingRequest);
      setTrainingMetrics(response);
      setSuccess(`Model trained successfully! Training score: ${(response.metrics.train_score * 100).toFixed(2)}%, Test score: ${(response.metrics.test_score * 100).toFixed(2)}%`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error training model');
    } finally {
      setIsTraining(false);
    }
  };

  const isStepValid = () => {
    switch (activeStep) {
      case 0:
        return !!selectedDataset;
      case 1:
        return !!selectedModel;
      case 2:
        return !!targetColumn && featureColumns.length > 0;
      default:
        return true;
    }
  };

  return (
    <Box sx={{ flexGrow: 1, p: 2 }}>
      <Paper sx={{ p: 3 }}>
        <Typography variant="h5" gutterBottom>
          Machine Learning Pipeline
        </Typography>
        <Stepper activeStep={activeStep} sx={{ my: 4 }}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        <Box sx={{ mt: 4 }}>
          {activeStep === 0 && (
            <FormControl fullWidth>
              <InputLabel>Select Dataset</InputLabel>
              <Select
                value={selectedDataset?.id || ''}
                label="Select Dataset"
                onChange={(e) => {
                  const dataset = datasets.find(d => d.id === e.target.value);
                  setSelectedDataset(dataset || null);
                }}
              >
                {datasets.map((dataset) => (
                  <MenuItem key={dataset.id} value={dataset.id}>
                    {dataset.name} ({dataset.shape[0]} rows, {dataset.shape[1]} columns)
                  </MenuItem>
                ))}
              </Select>
              {selectedDataset && (
                <FormHelperText>
                  Available columns: {selectedDataset.columns.join(', ')}
                </FormHelperText>
              )}
            </FormControl>
          )}

          {activeStep === 1 && (
            <FormControl fullWidth>
              <InputLabel>Select Model</InputLabel>
              <Select
                value={selectedModel}
                label="Select Model"
                onChange={(e) => setSelectedModel(e.target.value)}
              >
                <MenuItem value="linear_regression">Linear Regression</MenuItem>
                <MenuItem value="logistic_regression">Logistic Regression</MenuItem>
                <MenuItem value="random_forest">Random Forest</MenuItem>
                <MenuItem value="xgboost">XGBoost</MenuItem>
              </Select>
            </FormControl>
          )}

          {activeStep === 2 && selectedDataset && (
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <FormControl fullWidth>
                  <InputLabel>Target Column</InputLabel>
                  <Select
                    value={targetColumn}
                    label="Target Column"
                    onChange={(e) => setTargetColumn(e.target.value)}
                  >
                    {selectedDataset.columns.map((column) => (
                      <MenuItem key={column} value={column}>
                        {column}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12}>
                <FormControl fullWidth>
                  <InputLabel>Feature Columns</InputLabel>
                  <Select
                    multiple
                    value={featureColumns}
                    label="Feature Columns"
                    onChange={(e) => {
                      const value = e.target.value;
                      setFeatureColumns(typeof value === 'string' ? value.split(',') : value);
                    }}
                    renderValue={(selected) => (
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                        {selected.map((value) => (
                          <Chip key={value} label={value} />
                        ))}
                      </Box>
                    )}
                  >
                    {selectedDataset.columns
                      .filter(column => column !== targetColumn)
                      .map((column) => (
                        <MenuItem key={column} value={column}>
                          {column}
                        </MenuItem>
                      ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12}>
                <Typography gutterBottom>Test Size</Typography>
                <Slider
                  value={testSize}
                  onChange={(_, value) => setTestSize(value as number)}
                  min={0.1}
                  max={0.4}
                  step={0.05}
                  marks
                  valueLabelDisplay="auto"
                  valueLabelFormat={(value) => `${value * 100}%`}
                />
              </Grid>
              {(selectedModel === 'xgboost' || selectedModel === 'random_forest') && (
                <>
                  <Grid item xs={12}>
                    <Typography gutterBottom>Number of Epochs</Typography>
                    <Slider
                      value={epochs}
                      onChange={(_, value) => setEpochs(value as number)}
                      min={1}
                      max={100}
                      valueLabelDisplay="auto"
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <Typography gutterBottom>Learning Rate</Typography>
                    <Slider
                      value={learningRate}
                      onChange={(_, value) => setLearningRate(value as number)}
                      min={0.001}
                      max={0.1}
                      step={0.001}
                      valueLabelDisplay="auto"
                    />
                  </Grid>
                </>
              )}
            </Grid>
          )}

          {activeStep === 3 && (
            <Box>
              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={12} sm={4}>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      Model Configuration
                    </Typography>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                      <Chip label={`Dataset: ${selectedDataset?.name}`} />
                      <Chip label={`Model: ${selectedModel}`} />
                      <Chip label={`Target: ${targetColumn}`} />
                      <Chip label={`Features: ${featureColumns.length}`} />
                      <Chip label={`Test Size: ${testSize * 100}%`} />
                    </Box>
                  </Paper>
                </Grid>
                <Grid item xs={12} sm={8}>
                  {trainingMetrics && (
                    <Paper sx={{ p: 2, height: 300 }}>
                      <Typography variant="subtitle1" gutterBottom>
                        Training Results
                      </Typography>
                      <Box sx={{ mt: 2 }}>
                        <Typography variant="body1">
                          Training Score: {(trainingMetrics.metrics.train_score * 100).toFixed(2)}%
                        </Typography>
                        <Typography variant="body1">
                          Test Score: {(trainingMetrics.metrics.test_score * 100).toFixed(2)}%
                        </Typography>
                        {trainingMetrics.feature_importance && (
                          <Box sx={{ mt: 2 }}>
                            <Typography variant="subtitle2" gutterBottom>
                              Feature Importance
                            </Typography>
                            {Object.entries(trainingMetrics.feature_importance).map(([feature, importance]) => (
                              <Box key={feature} sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                                <Typography variant="body2" sx={{ minWidth: 120 }}>
                                  {feature}:
                                </Typography>
                                <Box
                                  sx={{
                                    flexGrow: 1,
                                    height: 8,
                                    bgcolor: 'primary.main',
                                    borderRadius: 1,
                                    maxWidth: `${(importance as number) * 100}%`,
                                  }}
                                />
                                <Typography variant="body2">
                                  {((importance as number) * 100).toFixed(1)}%
                                </Typography>
                              </Box>
                            ))}
                          </Box>
                        )}
                      </Box>
                    </Paper>
                  )}
                </Grid>
              </Grid>
            </Box>
          )}

          <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 3 }}>
            <Button
              disabled={activeStep === 0}
              onClick={handleBack}
              sx={{ mr: 1 }}
            >
              Back
            </Button>
            {activeStep === steps.length - 1 ? (
              <Button
                variant="contained"
                onClick={handleTrain}
                disabled={isTraining || !isStepValid()}
                startIcon={isTraining ? <CircularProgress size={20} /> : null}
              >
                {isTraining ? 'Training...' : 'Start Training'}
              </Button>
            ) : (
              <Button
                variant="contained"
                onClick={handleNext}
                disabled={!isStepValid()}
              >
                Next
              </Button>
            )}
          </Box>
        </Box>
      </Paper>

      <Snackbar
        open={!!error}
        autoHideDuration={6000}
        onClose={() => setError(null)}
      >
        <Alert severity="error" onClose={() => setError(null)}>
          {error}
        </Alert>
      </Snackbar>

      <Snackbar
        open={!!success}
        autoHideDuration={6000}
        onClose={() => setSuccess(null)}
      >
        <Alert severity="success" onClose={() => setSuccess(null)}>
          {success}
        </Alert>
      </Snackbar>
    </Box>
  );
} 