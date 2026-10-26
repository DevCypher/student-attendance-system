
import cv2
import csv
import json
import numpy as np

from pathlib import Path
from datetime import datetime
from insightface.app import FaceAnalysis


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

DATASET_DIR = Path(r"C:\Mini\Dataset2")

# Separate output folder: original database stays safe
OUTPUT_DIR = Path(
    r"C:\Mini\Student_attendace_system\face_database_v2"
)

DATABASE_FILE = OUTPUT_DIR / "student_embeddings_v2.npz"
REPORT_FILE = OUTPUT_DIR / "generation_report.json"
FAILED_FILE = OUTPUT_DIR / "failed_images.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------
# 2. INITIALIZE INSIGHTFACE
# --------------------------------------------------

def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading InsightFace buffalo_l model...")

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))

    # We will use the recognition network directly because
    # Dataset2 already contains cropped face images.
    recognition_model = app.models.get("recognition")

    if recognition_model is None:
        raise RuntimeError(
            "InsightFace recognition model was not loaded."
        )

    if not hasattr(recognition_model, "get_feat"):
        raise RuntimeError(
            "Recognition model does not support get_feat()."
        )

    print("Recognition model loaded.")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Output:  {OUTPUT_DIR}")
    print()


    # --------------------------------------------------
    # 3. FIND STUDENT FOLDERS
    # --------------------------------------------------

    student_folders = sorted(
        folder for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    )

    if not student_folders:
        raise RuntimeError(
            "No student folders found inside Dataset2."
        )

    print(f"Student folders found: {len(student_folders)}")

    all_embeddings = []
    all_student_ids = []
    all_image_paths = []
    failures = []
    student_summary = {}


    # --------------------------------------------------
    # 4. GENERATE EMBEDDINGS
    # --------------------------------------------------

    for student_folder in student_folders:
        student_id = student_folder.name

        images = sorted(
            path for path in student_folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        successful = 0
        failed = 0

        print(
            f"\nProcessing {student_id}: "
            f"{len(images)} images"
        )

        for image_path in images:
            try:
                image = cv2.imread(str(image_path))

                if image is None:
                    raise ValueError("Image could not be read.")

                height, width = image.shape[:2]

                if height < 40 or width < 40:
                    raise ValueError(
                        f"Image too small: {width}x{height}"
                    )

                # The images are already cropped.
                # Extract features directly instead of running
                # face detection on each crop again.
                embedding = recognition_model.get_feat(image)
                embedding = np.asarray(
                    embedding, dtype=np.float32
                ).reshape(-1)

                if not np.all(np.isfinite(embedding)):
                    raise ValueError(
                        "Embedding contains invalid values."
                    )

                norm = np.linalg.norm(embedding)

                if norm < 1e-10:
                    raise ValueError(
                        "Embedding has zero magnitude."
                    )

                # Normalize for cosine-similarity comparison
                embedding = embedding / norm

                all_embeddings.append(embedding)
                all_student_ids.append(student_id)
                all_image_paths.append(str(image_path))

                successful += 1

            except Exception as error:
                failed += 1
                failures.append({
                    "student_id": student_id,
                    "image_path": str(image_path),
                    "error": str(error)
                })

        student_summary[student_id] = {
            "images_found": len(images),
            "embeddings_generated": successful,
            "failed_images": failed
        }

        print(
            f"  Successful: {successful} | Failed: {failed}"
        )


    # --------------------------------------------------
    # 5. SAVE EMBEDDING DATABASE
    # --------------------------------------------------

    if not all_embeddings:
        raise RuntimeError(
            "No embeddings were generated. "
            "Check the image files and error report."
        )

    embeddings_array = np.vstack(all_embeddings)
    student_ids_array = np.asarray(
        all_student_ids, dtype=str
    )
    image_paths_array = np.asarray(
        all_image_paths, dtype=str
    )

    np.savez_compressed(
        DATABASE_FILE,
        embeddings=embeddings_array,
        student_ids=student_ids_array,
        image_paths=image_paths_array
    )


    # --------------------------------------------------
    # 6. SAVE FAILURE CSV
    # --------------------------------------------------

    with open(
        FAILED_FILE, "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["student_id", "image_path", "error"]
        )
        writer.writeheader()
        writer.writerows(failures)


    # --------------------------------------------------
    # 7. SAVE SUMMARY REPORT
    # --------------------------------------------------

    report = {
        "generated_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "dataset_directory": str(DATASET_DIR),
        "database_file": str(DATABASE_FILE),
        "model": "InsightFace buffalo_l recognition model",
        "embedding_method": (
            "Direct feature extraction from pre-cropped images"
        ),
        "students_found": len(student_folders),
        "total_images_found": sum(
            item["images_found"]
            for item in student_summary.values()
        ),
        "embeddings_generated": len(all_embeddings),
        "failed_images": len(failures),
        "embedding_dimensions": int(
            embeddings_array.shape[1]
        ),
        "students": student_summary
    }

    with open(
        REPORT_FILE, "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 48)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 48)
    print(f"Students:    {len(student_folders)}")
    print(f"Images:      {report['total_images_found']}")
    print(f"Embeddings:  {len(all_embeddings)}")
    print(f"Failures:    {len(failures)}")
    print(f"Dimensions:  {embeddings_array.shape[1]}")
    print(f"\nDatabase: {DATABASE_FILE}")
    print(f"Report:   {REPORT_FILE}")
    print(f"Failures: {FAILED_FILE}")


if __name__ == "__main__":
    main()
