# ML-Ops Showcase on Financial Data

This project demonstrates the implementation of a fully functional ML-Ops pipeline. It includes a simple frontend built with HTML and JavaScript, and a backend using FastAPI, PyTorch, and Plotly. The pipeline allows users to select a publicly traded asset, such as Apple stock ('AAPL') or Bitcoin ('BTC-USD'). It then looks up the Model_DB for trained models corresponding to the selected symbol and plots the last 60 closing prices along with the model's predictions.

Users can train new models on the data and delete existing models directly from the user interface. For security, the API requests are safeguarded, and the app refreshes the plot 10 times after the last prediction click. Additionally, the maximum number of models is capped at 10 in the database.

The project is designed for easy extension. To add a new model architecture, simply create a new model class and a corresponding `.yaml` file in the `modelclasses` folder. A template for the model class is provided.

## Running the Project Locally:

Using the `Dockerfile` and `requirements.txt`, you can create a Docker image by running:

\`\`\`bash
docker build -t <img_name> .
\`\`\`

Then, run the Docker container with:

\`\`\`bash
docker run -d -p 8000:8000 <img_name>
\`\`\`

To access the project on your local machine, go to: [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Deploying the Docker Image on an EC2 Instance:

1. Log in as an IAM user:

    \`\`\`bash
    aws configure
    \`\`\`

2. Log in to ECR:

    \`\`\`bash
    aws ecr get-login-password --region <aws_region> | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.<aws_region>.amazonaws.com
    \`\`\`

3. Tag and push the Docker image to the ECR repository:

    \`\`\`bash
    docker tag <docker_image_name>:latest <aws_account_id>.dkr.ecr.<aws_region>.amazonaws.com/<repository_name>:latest
    docker push <aws_account_id>.dkr.ecr.<aws_region>.amazonaws.com/<repository_name>:latest
    \`\`\`

4. Create and launch an EC2 instance on AWS.

5. Log into the EC2 instance using SSH:

    \`\`\`bash
    ssh -i <Name.pem> ec2-user@<ipv4_address>
    \`\`\`

6. Log in to ECR on the EC2 instance, pull the Docker image, and run it:

    \`\`\`bash
    sudo aws ecr get-login-password --region <aws_region> | sudo docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.<aws_region>.amazonaws.com
    sudo docker pull <aws_account_id>.dkr.ecr.<aws_region>.amazonaws.com/<repository_name>:latest
    sudo docker run -d --name <container_name> -p <port>:<port> <aws_account_id>.dkr.ecr.<aws_region>.amazonaws.com/<repository_name>:latest
    \`\`\`