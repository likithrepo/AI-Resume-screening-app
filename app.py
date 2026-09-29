import os
from functools import wraps
from flask import (Flask, render_template, request, redirect, url_for,
                    session, flash, abort)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from db import get_db, init_db
from matching import compute_match_score
from loaders import load_text_from_file, is_allowed_file
import groq_client

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads", "resumes")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("JOB_PORTAL_SECRET_KEY", "dev-secret-change-me")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB resume upload cap


# --------------------------------------------------------------------- auth
def login_required(role=None):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if "user_id" not in session:
                flash("Please log in first.", "error")
                return redirect(url_for("login"))
            if role and session.get("role") != role:
                abort(403)
            return fn(*args, **kwargs)
        return wrapper
    return decorator


@app.route("/")
def index():
    if "user_id" not in session:
        return redirect(url_for("login"))
    if session["role"] == "recruiter":
        return redirect(url_for("recruiter_dashboard"))
    return redirect(url_for("job_list"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        role = request.form["role"]

        if role not in ("recruiter", "job_seeker"):
            flash("Invalid role selected.", "error")
            return redirect(url_for("register"))
        if not name or not email or not password:
            flash("All fields are required.", "error")
            return redirect(url_for("register"))

        db = get_db()
        existing = db.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            flash("An account with that email already exists.", "error")
            db.close()
            return redirect(url_for("register"))

        db.execute(
            "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)",
            (name, email, generate_password_hash(password), role),
        )
        db.commit()
        db.close()
        flash("Account created. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        db = get_db()
        user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        db.close()

        if user is None or not check_password_hash(user["password_hash"], password):
            flash("Invalid email or password.", "error")
            return redirect(url_for("login"))

        session["user_id"] = user["id"]
        session["name"] = user["name"]
        session["role"] = user["role"]
        return redirect(url_for("index"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------------------- recruiter
@app.route("/recruiter/dashboard")
@login_required(role="recruiter")
def recruiter_dashboard():
    db = get_db()
    jobs = db.execute(
        """SELECT jobs.*, COUNT(applications.id) AS applicant_count
           FROM jobs LEFT JOIN applications ON applications.job_id = jobs.id
           WHERE jobs.recruiter_id = ?
           GROUP BY jobs.id ORDER BY jobs.created_at DESC""",
        (session["user_id"],),
    ).fetchall()
    db.close()
    return render_template("recruiter_dashboard.html", jobs=jobs)


@app.route("/recruiter/jobs/new", methods=["GET", "POST"])
@login_required(role="recruiter")
def post_job():
    if request.method == "POST":
        title = request.form["title"].strip()
        company = request.form["company"].strip()
        description = request.form["description"].strip()
        if not title or not description:
            flash("Title and description are required.", "error")
            return redirect(url_for("post_job"))

        db = get_db()
        db.execute(
            "INSERT INTO jobs (recruiter_id, title, company, description) VALUES (?, ?, ?, ?)",
            (session["user_id"], title, company, description),
        )
        db.commit()
        db.close()
        flash("Job posted.", "success")
        return redirect(url_for("recruiter_dashboard"))

    return render_template("post_job.html")


@app.route("/recruiter/jobs/<int:job_id>")
@login_required(role="recruiter")
def view_applicants(job_id):
    db = get_db()
    job = db.execute(
        "SELECT * FROM jobs WHERE id = ? AND recruiter_id = ?",
        (job_id, session["user_id"]),
    ).fetchone()
    if job is None:
        db.close()
        abort(404)

    applicants = db.execute(
        """SELECT applications.*, users.name AS seeker_name, users.email AS seeker_email
           FROM applications JOIN users ON users.id = applications.seeker_id
           WHERE applications.job_id = ?
           ORDER BY applications.match_score DESC""",
        (job_id,),
    ).fetchall()
    db.close()

    applicants = [
        dict(
            a,
            matched_keywords=(a["matched_keywords"] or "").split(",") if a["matched_keywords"] else [],
            llm_strengths=(a["llm_strengths"] or "").split("\n") if a["llm_strengths"] else [],
            llm_gaps=(a["llm_gaps"] or "").split("\n") if a["llm_gaps"] else [],
        )
        for a in applicants
    ]
    return render_template("view_applicants.html", job=job, applicants=applicants)


# --------------------------------------------------------------- job seeker
@app.route("/jobs")
@login_required(role="job_seeker")
def job_list():
    db = get_db()
    jobs = db.execute(
        """SELECT jobs.*, users.name AS recruiter_name,
                  (SELECT COUNT(*) FROM applications
                   WHERE applications.job_id = jobs.id AND applications.seeker_id = ?) AS already_applied
           FROM jobs JOIN users ON users.id = jobs.recruiter_id
           ORDER BY jobs.created_at DESC""",
        (session["user_id"],),
    ).fetchall()
    db.close()
    return render_template("job_list.html", jobs=jobs)


@app.route("/jobs/<int:job_id>", methods=["GET", "POST"])
@login_required(role="job_seeker")
def job_detail(job_id):
    db = get_db()
    job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if job is None:
        db.close()
        abort(404)

    already_applied = db.execute(
        "SELECT id FROM applications WHERE job_id = ? AND seeker_id = ?",
        (job_id, session["user_id"]),
    ).fetchone()

    if request.method == "POST":
        if already_applied:
            flash("You've already applied to this job.", "error")
            db.close()
            return redirect(url_for("job_detail", job_id=job_id))

        resume_text = ""
        resume_filename = None

        pasted = request.form.get("resume_text", "").strip()
        file = request.files.get("resume_file")

        if file and file.filename:
            if not is_allowed_file(file.filename):
                flash("Unsupported resume file type. Use .txt, .pdf, or .docx.", "error")
                db.close()
                return redirect(url_for("job_detail", job_id=job_id))
            filename = secure_filename(f"{session['user_id']}_{job_id}_{file.filename}")
            path = os.path.join(UPLOAD_DIR, filename)
            file.save(path)
            resume_filename = filename
            try:
                resume_text = load_text_from_file(path)
            except Exception as e:
                flash(f"Could not read resume file: {e}", "error")
                db.close()
                return redirect(url_for("job_detail", job_id=job_id))
        elif pasted:
            resume_text = pasted
        else:
            flash("Upload a resume file or paste your resume text.", "error")
            db.close()
            return redirect(url_for("job_detail", job_id=job_id))

        score, matched_keywords = compute_match_score(job["description"], resume_text)
        analysis = groq_client.analyze_application(job["description"], resume_text)

        db.execute(
            """INSERT INTO applications
               (job_id, seeker_id, resume_filename, resume_text, match_score, matched_keywords,
                llm_score, llm_summary, llm_strengths, llm_gaps)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job_id, session["user_id"], resume_filename, resume_text, score, ",".join(matched_keywords),
                analysis["llm_score"] if analysis else None,
                analysis["summary"] if analysis else None,
                "\n".join(analysis["strengths"]) if analysis else None,
                "\n".join(analysis["gaps"]) if analysis else None,
            ),
        )
        db.commit()
        db.close()
        flash("Application submitted!", "success")
        return redirect(url_for("job_list"))

    db.close()
    return render_template("job_detail.html", job=job, already_applied=already_applied)


@app.route("/seeker/dashboard")
@login_required(role="job_seeker")
def seeker_dashboard():
    db = get_db()
    applications = db.execute(
        """SELECT applications.*, jobs.title, jobs.company
           FROM applications JOIN jobs ON jobs.id = applications.job_id
           WHERE applications.seeker_id = ?
           ORDER BY applications.applied_at DESC""",
        (session["user_id"],),
    ).fetchall()
    db.close()
    return render_template("seeker_dashboard.html", applications=applications)


if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)
