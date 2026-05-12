pipeline {
    agent any

    stages {

        stage('Clone Repository') {
            steps {
                git branch: 'devops-integration',
                url: 'https://github.com/ARCHI-DUBEY/AQI-PULSE.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                bat 'docker build -t aqi-pulse .'
            }
        }

        stage('Stop Old Container') {
            steps {
                bat 'docker stop aqi-container || exit 0'
                bat 'docker rm aqi-container || exit 0'
            }
        }

        stage('Run Docker Container') {
            steps {
                bat 'bat docker run -d --name aqi-container --env-file .env -p 8501:8501 aqi-pulse'
            }
        }
    }
}