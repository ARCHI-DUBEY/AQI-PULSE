pipeline {
    agent any

    stages {

        stage('Clone Repository') {
            steps {
                echo 'Cloning GitHub repository...'
            }
        }

        stage('Build Docker Image') {
            steps {
                bat 'docker build -t aqi-pulse .'
            }
        }

        stage('Run Docker Container') {
            steps {
                bat 'docker stop aqi-container || exit 0'
                bat 'docker rm aqi-container || exit 0'
                bat 'docker run -d -p 8501:8501 --name aqi-container aqi-pulse'
            }
        }

    }
}