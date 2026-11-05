# Heart Disease Prediction API

FastAPI serves a scikit-learn random forest trained on the Kaggle Heart Disease Dataset. This project is for a Docker deployment assignment and is intended for educational use.

## Run locally with Docker

Start Docker Desktop (or Docker Engine with Compose) and run these commands from the repository root.

```sh
docker compose build
docker compose up
```

Open [Swagger UI](http://localhost:8000/docs). To stop the service, press Ctrl+C and run `docker compose down`. The image includes `model/heart_model.joblib` and loads it on startup. It uses port `8000` by default and honors the `PORT` environment variable on Render.

## Test the API

| Method and path | Response |
| --- | --- |
| `GET /health` | `{"status":"ok"}` once the model has loaded |
| `GET /info` | Classifier type and the ordered list of 13 input features |
| `POST /predict` | `{"heart_disease":true}` for predicted presence or `{"heart_disease":false}` for predicted absence |

```powershell
$base = 'http://localhost:8000'
curl.exe --fail-with-body "$base/health"
curl.exe --fail-with-body "$base/info"
curl.exe --fail-with-body "$base/predict" -H 'Content-Type: application/json' --data-binary '@examples/prediction.json'
```

The example returns `{"heart_disease":true}` with the committed model. Missing fields, unexpected fields, and values outside the schema's bounds return HTTP 422. The schema and field descriptions are available in Swagger UI.

The categorical inputs use the Kaggle CSV codes. `cp` uses `0` for asymptomatic, `1` for atypical angina, `2` for non-anginal pain, and `3` for typical angina. `restecg` uses `0` for probable or definite left ventricular hypertrophy, `1` for normal, and `2` for ST–T wave abnormality. `slope` uses `0` for downsloping, `1` for flat, and `2` for upsloping. `ca` accepts counts `0`–`3` and uses `4` for missing. `thal` uses `0` for missing, `1` for fixed defect, `2` for normal, and `3` for reversible defect.

## Reproduce the model

Download and extract `heart.csv` into a `data/` directory at the repository root. The CSV used for training has SHA-256 `ddb2996b2f4db2e00aad13f4518200179ff69f79093838e3c21ffa672ebec0f1`. [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

With Python 3.13, run the following in PowerShell. The training command updates `model/heart_model.joblib`.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m model.train data/heart.csv
```

The downloaded CSV has 1,025 rows. Matching them to the original Cleveland records on age, sex, resting blood pressure, cholesterol, fasting blood sugar, maximum heart rate, exercise-induced angina, and ST depression showed that all 499 rows with Kaggle `target=0` correspond to Cleveland `num>0` (disease present). All 526 rows with Kaggle `target=1` correspond to Cleveland `num=0` (disease absent). The trainer converts the labels to `1` for presence and `0` for absence. [UCI dataset documentation](https://archive.ics.uci.edu/dataset/45/heart+disease) · [Original Cleveland CSV](https://archive.ics.uci.edu/static/public/45/data.csv)

The trainer removes duplicate feature rows before making a stratified 80/20 split with seed `42`. The committed random forest achieved 78.7% accuracy on 61 held-out records, with 241 records used for training.

To run the API directly with the same virtual environment, use `.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000`.

## Deploy to Render

1. Create an empty GitHub repository and push the local `main` branch. Replace `YOUR-USERNAME` with your GitHub username and `YOUR-REPO` with your repository name.

   ```sh
   git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
   git push -u origin main
   ```

2. In Render, create a **Web Service** from that repository and select `main`.
3. Set **Language** to **Docker**. Leave **Root Directory** blank, use `.` for **Docker Build Context Directory**, and keep **Dockerfile Path** at `./Dockerfile`.
4. Set **Health Check Path** to `/health`. The Dockerfile starts Uvicorn on `0.0.0.0` and reads Render's `PORT` variable. [Docker on Render](https://render.com/docs/docker) · [Render web services](https://render.com/docs/web-services)
5. Deploy and wait for Render to report that the service is live. Repeat the three API checks above with `$base` set to the assigned `https://…onrender.com` URL, then open `/docs` on that URL.
6. Submit the GitHub repository URL and the live Render URL.
