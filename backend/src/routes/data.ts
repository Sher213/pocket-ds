import express from 'express';
import multer from 'multer';
import path from 'path';
import { z } from 'zod';
import { Request, Response } from 'express';
import { PythonShell } from 'python-shell';

const router = express.Router();

// Configure multer for dataset uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/datasets/');
  },
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`);
  },
});

const upload = multer({
  storage,
  fileFilter: (req, file, cb) => {
    const allowedExtensions = ['.csv', '.xlsx', '.json'];
    const ext = path.extname(file.originalname).toLowerCase();
    if (allowedExtensions.includes(ext)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Only CSV, Excel, and JSON files are allowed.'));
    }
  },
});

// Schema for dataset metadata
const datasetMetadataSchema = z.object({
  name: z.string().min(1),
  description: z.string().optional(),
  tags: z.array(z.string()).optional(),
});

// Routes
router.post(
  '/upload',
  upload.single('file'),
  async (req: Request, res: Response): Promise<void> => {
    try {
      if (!req.file) {
        res.status(400).json({ error: 'No file uploaded' });
        return;
      }

      const metadata = datasetMetadataSchema.parse(req.body);

      // TODO: Process the uploaded file and store metadata in the database
      const fileInfo = {
        filename: req.file.filename,
        originalName: req.file.originalname,
        size: req.file.size,
        path: req.file.path,
        ...metadata,
      };

      res.json({
        message: 'File uploaded successfully',
        dataset: fileInfo,
      });
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ error: 'Invalid metadata', details: error.errors });
      } else {
        res.status(500).json({ error: 'Error uploading file' });
      }
    }
  }
);

router.get('/list', async (req, res) => {
  try {
    // TODO: Fetch datasets from database
    const mockDatasets = [
      {
        id: '1',
        name: 'Housing Prices',
        description: 'Dataset containing house prices and features',
        filename: 'housing.csv',
        uploadDate: '2024-03-15',
        size: 1024576,
      },
      // Add more mock datasets
    ];

    res.json({ datasets: mockDatasets });
  } catch (error) {
    res.status(500).json({ error: 'Error fetching datasets' });
  }
});

router.get('/:id/preview', async (req, res) => {
  try {
    const { id } = req.params;
    const { rows = 10 } = req.query;

    // TODO: Implement actual data preview logic
    const mockPreview = {
      columns: ['price', 'bedrooms', 'bathrooms', 'sqft'],
      rows: [
        { price: 300000, bedrooms: 3, bathrooms: 2, sqft: 1500 },
        { price: 400000, bedrooms: 4, bathrooms: 2.5, sqft: 2000 },
        // Add more mock rows
      ],
      totalRows: 1000,
    };

    res.json(mockPreview);
  } catch (error) {
    res.status(500).json({ error: 'Error generating preview' });
  }
});

router.delete('/:id', async (req, res) => {
  try {
    const { id } = req.params;

    // TODO: Implement actual dataset deletion logic
    res.json({ message: 'Dataset deleted successfully' });
  } catch (error) {
    res.status(500).json({ error: 'Error deleting dataset' });
  }
});

export const dataRoutes = router; 