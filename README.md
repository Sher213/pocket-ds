# Pocket Data Scientist

A comprehensive web application that brings the power of data science to your browser. This application enables users to perform data analysis, visualization, machine learning, and reporting seamlessly.

## Features

- 📊 **Data Import & Handling**: Support for multiple formats (CSV, Excel, JSON, SQL)
- 🔍 **Exploratory Data Analysis**: Interactive visualizations and statistical analysis
- 🤖 **Machine Learning**: Pre-built models and AutoML capabilities
- 📝 **Reporting**: Generate comprehensive analysis reports
- 👥 **Collaboration**: Share projects and results with team members
- 📚 **Learning Resources**: Integrated tutorials and documentation

## Tech Stack

- **Frontend**: React.js with TypeScript
- **Backend**: Node.js with Express
- **Database**: PostgreSQL
- **Machine Learning**: Python with scikit-learn, TensorFlow
- **Authentication**: JWT-based auth system
- **Cloud Services**: AWS/Azure for deployment

## Getting Started

### Prerequisites

- Node.js (v18 or higher)
- Python 3.8+
- PostgreSQL 13+
- npm or yarn

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pocket-data-scientist.git
cd pocket-data-scientist
```

2. Install frontend dependencies:
```bash
cd frontend
npm install
```

3. Install backend dependencies:
```bash
cd ../backend
npm install
```

4. Set up Python environment and dependencies:
```bash
cd ../ml-service
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

5. Configure environment variables:
- Copy `.env.example` to `.env` in both frontend and backend directories
- Update the variables with your configuration

6. Start the development servers:
```bash
# Terminal 1 - Frontend
cd frontend
npm run dev

# Terminal 2 - Backend
cd backend
npm run dev

# Terminal 3 - ML Service
cd ml-service
python app.py
```

## Project Structure

```
pocket-data-scientist/
├── frontend/           # React frontend application
├── backend/           # Node.js/Express backend server
├── ml-service/        # Python ML service
└── docs/             # Documentation
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with ❤️ for data scientists and analysts
- Powered by open-source machine learning libraries
- Inspired by the need for accessible data science tools 