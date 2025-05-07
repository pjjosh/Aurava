// frontend/src/components/EmotionMusic.jsx
import React, { useRef, useState, useEffect } from "react";
import axios from "axios";
import "./EmotionMusic.css";

export default function EmotionMusic() {
  const videoRef = useRef();
  const canvasRef = useRef();

  const [emotion, setEmotion] = useState("");
  const [genre, setGenre] = useState("classical");
  const [audioUrl, setAudioUrl] = useState("");
  const [loadingEmotion, setLoadingEmotion] = useState(false);
  const [loadingMusic, setLoadingMusic] = useState(false);
  const [error, setError] = useState("");
  const [debugImage, setDebugImage] = useState(null);
  const NUM_NOTES = 25;
  const [notes, setNotes] = useState([]);

  const NOTE_SVG = 
  "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E" +
  "%3Cpath fill='%23fff' d='M12 3v10.55A4 4 0 1014 17V7h4V3z'/%3E" +
  "%3C/svg%3E";
  useEffect(() => {
    const arr = Array.from({ length: NUM_NOTES }).map(() => ({
      top:  Math.random() * 100,           // percent
      left: Math.random() * 100,           // percent
      size: 20 + Math.random() * 60,       // px
      rot:  Math.random() * 360,           // degrees
      op:   0.1 + Math.random() * 0.3,     // low opacity
      key:  Math.random().toString(36).slice(2),
    }));
    setNotes(arr);
    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoRef.current.srcObject = stream;
      } catch {
        setError("Unable to access webcam.");
      }
    }
    startCamera();
    return () => {
      if (videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  const captureEmotion = async () => {
    setError("");
    setLoadingEmotion(true);
    try {
      const ctx = canvasRef.current.getContext("2d");
      ctx.drawImage(videoRef.current, 0, 0, 224, 224);

      // ==== DEBUG: grab the exact JPEG we send ====
      const dataUrl = canvasRef.current.toDataURL("image/jpeg");
      setDebugImage(dataUrl);

      const blob = await new Promise((res) =>
        canvasRef.current.toBlob(res, "image/jpeg")
      );
      const form = new FormData();
      form.append("file", blob, "snapshot.jpg");

      const { data } = await axios.post(
        "http://localhost:8000/predict_emotion",
        form,
        { headers: { "Content-Type": "multipart/form-data" } }
      );
      setEmotion(data.emotion);
    } catch {
      setError("Failed to detect emotion.");
    } finally {
      setLoadingEmotion(false);
    }
  };

  const generateMusic = async () => {
    setError("");
    setLoadingMusic(true);
    try {
      const { data } = await axios.post(
        "http://localhost:8000/generate_music",
        { emotion, genre },
        { headers: { "Content-Type": "application/json" } }
      );
      setAudioUrl(`http://localhost:8000${data.url}`);
    } catch {
      setError("Music generation failed.");
    } finally {
      setLoadingMusic(false);
    }
  };

  return (
    <div className="emm-app">
      {notes.map(n => (
      <div
        key={n.key}
        style={{
        position:       "absolute",
        top:            `${n.top}%`,
        left:           `${n.left}%`,
        width:          `${n.size}px`,
        height:         `${n.size}px`,
        backgroundImage:`url("${NOTE_SVG}")`,
        backgroundSize: "contain",
        backgroundRepeat:"no-repeat",
        transform:      `rotate(${n.rot}deg)`,
        opacity:        n.op,
        pointerEvents:  "none",
        }}
        />
      ))}
      <div className="emm-card">
        <h2 className="emm-title">Aurava</h2>

        {error && <p className="emm-error">{error}</p>}

        <div className="emm-video-container">
          <video
            ref={videoRef}
            width={224}
            height={224}
            autoPlay
            muted
            className="emm-video"
          />
          <button
            onClick={captureEmotion}
            disabled={loadingEmotion}
            className="emm-detect-btn"
          >
            {loadingEmotion ? "Detecting…" : "Detect Emotion"}
          </button>
        </div>

        {/* ==== DEBUG: show what we sent to the server ==== */}
        {debugImage && (
          <div style={{ margin: "1rem 0", textAlign: "center" }}>
            <p style={{ marginBottom: "0.5rem", color: "#444" }}>
              Snapshot sent:
            </p>
            <img
              src={debugImage}
              width={224}
              height={224}
              alt="debug snapshot"
              style={{ border: "2px solid #666", borderRadius: "4px" }}
            />
          </div>
        )}

        {emotion && <p className="emm-emotion">Detected: {emotion}</p>}

        <div>
          <label htmlFor="genre" className="emm-genre-label">
            Genre:
          </label>
          <select
            id="genre"
            value={genre}
            onChange={(e) => setGenre(e.target.value)}
            className="emm-genre-select"
          >
            <option value="classical">Classical</option>
            <option value="jazz">Jazz</option>
            <option value="rock">Rock</option>
            <option value="edm">EDM</option>
            <option value="ambient">Ambient</option>
            <option value="pop">Pop</option>
            <option value="hiphop">Hip-Hop</option>
            <option value="country">Country</option>
            <option value="reggae">Reggae</option>
            <option value="metal">Metal</option>
            <option value="r&b">R&B</option>
            <option value="funk">Funk</option>
          </select>
        </div>

        <button
          onClick={generateMusic}
          disabled={!emotion || loadingMusic}
          className="emm-generate-btn"
        >
          {loadingMusic ? "Creating…" : "Generate Music"}
        </button>

        {audioUrl && (
          <audio src={audioUrl} controls className="emm-audio" />
        )}

        {/* hidden canvas for snapshot */}
        <canvas
          ref={canvasRef}
          width={224}
          height={224}
          style={{ display: "none" }}
        />
      </div>
    </div>
);
}
