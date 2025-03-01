// src/components/UploadForm.js
import React, { useState } from "react";
import { uploadAudio } from "../api";

const UploadForm = () => {
    const [file, setFile] = useState(null);
    const [result, setResult] = useState(null);
    const [error, setError] = useState("");

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
    };

    const handleUpload = async (e) => {
        e.preventDefault();
        if (!file) {
            setError("Please select a file.");
            return;
        }
        setError("");
        const formData = new FormData();
        formData.append("file", file);
        try {
            const data = await uploadAudio(formData);
            if (data.error) {
                setError(data.error);
            } else {
                setResult(data);
            }
        } catch (err) {
            setError("Failed to process audio.");
            console.error(err);
        }
    };

    return (
        <div>
            <h2>Upload Your Audio</h2>
            <form onSubmit={handleUpload}>
                <input type="file" onChange={handleFileChange} accept=".mp3, .wav" />
                <button type="submit">Upload & Analyze</button>
            </form>
            {error && <p style={{ color: "red" }}>{error}</p>}
            {result && (
                <div>
                    <h1>File received: {}</h1>
                    <h2>Result: {result.result_label}</h2>
                    <h3>Fake Probability: {result.fake_probability.toFixed(2)}%</h3>
                    <h3>Real Probability: {result.real_probability.toFixed(2)}%</h3>
                </div>
            )}
        </div>
    );
};

export default UploadForm;
