import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:4000';
const ML_API_URL = import.meta.env.VITE_ML_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
});

const mlApi = axios.create({
  baseURL: ML_API_URL,
});

export interface DatasetMetadata {
  name: string;
  description?: string;
  tags?: string[];
}

export interface UploadResponse {
  dataset_id: string;
  message: string;
  shape: [number, number];
  columns: string[];
  preview: Record<string, any>[];
}

export interface TrainingRequest {
  dataset_id: string;
  model_type: 'linear_regression' | 'logistic_regression' | 'random_forest' | 'xgboost';
  target_column: string;
  feature_columns: string[];
  test_size?: number;
  random_state?: number;
  hyperparameters?: Record<string, any>;
}

export interface TrainingResponse {
  model_id: string;
  metrics: {
    train_score: number;
    test_score: number;
  };
  feature_importance?: Record<string, number>;
}

export interface ModelDetails {
  id: string;
  type: string;
  created: string;
  metrics: {
    train_score: number;
    test_score: number;
    rmse?: number;
    mae?: number;
    r2?: number;
  };
  hyperparameters: Record<string, any>;
  feature_importance?: Record<string, number>;
  dataset: string;
  last_used?: string;
  predictions_count?: number;
}

export const dataService = {
  async uploadDataset(file: File, metadata: DatasetMetadata): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file); // Ensure the file is correctly added
    formData.append('name', metadata.name);
    formData.append('description', metadata.description || ' ');

    try {
      const response = await mlApi.post('/dataset/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(`Upload failed: ${error.response?.statusText || error.message}`);
      } else {
        throw new Error('Error uploading dataset');
      }
    }
  },

  async listDatasets() {
    const response = await mlApi.get('/dataset/list');
    return response.data;
  },

  async previewDataset(datasetId: string, rows: number = 10) {
    const response = await mlApi.get(`/dataset/${datasetId}/preview`, {
      params: { rows },
    });
    return response.data;
  },

  async deleteDataset(datasetId: string) {
    const response = await mlApi.delete(`/dataset/${datasetId}`);
    return response.data;
  },
};

export const mlService = {
  async trainModel(request: TrainingRequest): Promise<TrainingResponse> {
    const response = await mlApi.post<TrainingResponse>('/train', request);
    return response.data;
  },

  async listModels() {
    const response = await mlApi.get('/models');
    return response.data;
  },

  async getModelDetails(modelId: string): Promise<ModelDetails> {
    const response = await mlApi.get(`/model/${modelId}`);
    return response.data;
  },

  async deleteModel(modelId: string) {
    const response = await mlApi.delete(`/model/${modelId}`);
    return response.data;
  },

  async predict(modelId: string, data: Record<string, any>[]) {
    const response = await mlApi.post(`/predict/${modelId}`, { data });
    return response.data;
  },
}; 