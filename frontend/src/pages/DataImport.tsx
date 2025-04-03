import { useState, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Alert,
  Snackbar,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  Description as FileIcon,
  Delete as DeleteIcon,
} from '@mui/icons-material';
import { dataService, type DatasetMetadata, type UploadResponse } from '../services/api';

interface FileWithPreview extends File {
  preview?: string;
}

interface UploadDialogProps {
  open: boolean;
  onClose: () => void;
  onConfirm: (metadata: DatasetMetadata) => void;
  loading: boolean;
}

const UploadDialog = ({ open, onClose, onConfirm, loading }: UploadDialogProps) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');

  const handleConfirm = () => {
    onConfirm({
      name,
      description: description || undefined,
    });
    setName('');
    setDescription('');
  };

  return (
    <Dialog
      open={open}
      onClose={() => {
        setName('');
        setDescription('');
        onClose();
      }}
    >
      <DialogTitle>Dataset Details</DialogTitle>
      <DialogContent>
        <Box sx={{ pt: 2 }}>
          <TextField
            autoFocus
            margin="dense"
            label="Dataset Name"
            fullWidth
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
            sx={{ mb: 2 }}
          />
          <TextField
            margin="dense"
            label="Description"
            fullWidth
            multiline
            rows={3}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
          />
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} disabled={loading}>Cancel</Button>
        <Button
          onClick={handleConfirm}
          variant="contained"
          disabled={loading || !name}
        >
          {loading ? <CircularProgress size={24} /> : 'Upload'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default function DataImport() {
  const [files, setFiles] = useState<FileWithPreview[]>([]);
  const [uploading, setUploading] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [currentFile, setCurrentFile] = useState<FileWithPreview | null>(null);
  const [uploadResponse, setUploadResponse] = useState<UploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: FileWithPreview[]) => {
    setFiles((prevFiles) => [...prevFiles, ...acceptedFiles]);
  }, []);

  const removeFile = (index: number) => {
    setFiles((prevFiles) => prevFiles.filter((_, i) => i !== index));
  };

  const handleUpload = async (metadata: DatasetMetadata) => {
    if (!currentFile) return;

    setUploading(true);
    setError(null);

    try {
      const response = await dataService.uploadDataset(currentFile, metadata);
      setUploadResponse(response);
      setFiles((prevFiles) => prevFiles.filter(f => f !== currentFile));
      setDialogOpen(false);
      setCurrentFile(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error uploading file');
    } finally {
      setUploading(false);
    }
  };

  const initiateUpload = (file: FileWithPreview) => {
    setCurrentFile(file);
    setDialogOpen(true);
  };

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto', p: 2 }}>
      <Paper
        sx={{
          p: 3,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          backgroundColor: (theme) => theme.palette.grey[50],
          border: '2px dashed',
          borderColor: 'primary.main',
          cursor: 'pointer',
          '&:hover': {
            borderColor: 'primary.dark',
            backgroundColor: (theme) => theme.palette.grey[100],
          },
        }}
        onDrop={(e) => {
          e.preventDefault();
          const droppedFiles = Array.from(e.dataTransfer.files);
          onDrop(droppedFiles as FileWithPreview[]);
        }}
        onDragOver={(e) => {
          e.preventDefault();
        }}
      >
        <UploadIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
        <Typography variant="h6" gutterBottom>
          Drag and drop files here
        </Typography>
        <Typography variant="body2" color="text.secondary" align="center">
          or
        </Typography>
        <Button
          component="label"
          variant="contained"
          sx={{ mt: 2 }}
          startIcon={<UploadIcon />}
        >
          Browse Files
          <input
            type="file"
            hidden
            multiple
            onChange={(e) => {
              if (e.target.files) {
                onDrop(Array.from(e.target.files) as FileWithPreview[]);
              }
            }}
          />
        </Button>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
          Supported formats: CSV, Excel, JSON
        </Typography>
      </Paper>

      {files.length > 0 && (
        <Paper sx={{ mt: 3, p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Selected Files
          </Typography>
          <List>
            {files.map((file, index) => (
              <ListItem
                key={index}
                secondaryAction={
                  <>
                    <Button
                      onClick={() => initiateUpload(file)}
                      disabled={uploading}
                      sx={{ mr: 1 }}
                    >
                      Upload
                    </Button>
                    <IconButton edge="end" onClick={() => removeFile(index)}>
                      <DeleteIcon />
                    </IconButton>
                  </>
                }
              >
                <ListItemIcon>
                  <FileIcon />
                </ListItemIcon>
                <ListItemText
                  primary={file.name}
                  secondary={`${(file.size / 1024 / 1024).toFixed(2)} MB`}
                />
              </ListItem>
            ))}
          </List>
        </Paper>
      )}

      <UploadDialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        onConfirm={handleUpload}
        loading={uploading}
      />

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
        open={!!uploadResponse}
        autoHideDuration={6000}
        onClose={() => setUploadResponse(null)}
      >
        <Alert severity="success" onClose={() => setUploadResponse(null)}>
          File uploaded successfully! Columns: {uploadResponse?.columns.length}, Rows: {uploadResponse?.shape[0]}
        </Alert>
      </Snackbar>
    </Box>
  );
}
