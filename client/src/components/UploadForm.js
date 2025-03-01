// // src/components/UploadForm.js
// import React, { useState } from "react";
// import { uploadAudio } from "../api";

// const UploadForm = () => {
//     const [file, setFile] = useState(null);
//     const [result, setResult] = useState(null);
//     const [error, setError] = useState("");

//     const handleFileChange = (e) => {
//         setFile(e.target.files[0]);
//     };

//     const handleUpload = async (e) => {
//         e.preventDefault();
//         if (!file) {
//             setError("Please select a file.");
//             return;
//         }
//         setError("");
//         const formData = new FormData();
//         formData.append("file", file);
//         try {
//             const data = await uploadAudio(formData);
//             if (data.error) {
//                 setError(data.error);
//             } else {
//                 setResult(data);
//             }
//         } catch (err) {
//             setError("Failed to process audio.");
//             console.error(err);
//         }
//     };

//     return (
//         <div>
//             <h2>Upload Your Audio</h2>
//             <form onSubmit={handleUpload}>
//                 <input type="file" onChange={handleFileChange} accept=".mp3, .wav" />
//                 <button type="submit">Upload & Analyze</button>
//             </form>
//             {error && <p style={{ color: "red" }}>{error}</p>}
//             {result && (
//                 <div>
//                     <h1>File received: {}</h1>
//                     <h2>Result: {result.result_label}</h2>
//                     <h3>Fake Probability: {result.fake_probability.toFixed(2)}%</h3>
//                     <h3>Real Probability: {result.real_probability.toFixed(2)}%</h3>
//                 </div>
//             )}
//         </div>
//     );
// };

// export default UploadForm;

// src/components/UploadForm.js
// src/components/UploadForm.js
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { uploadAudio } from "../api"; // Ensure your API call function is set up
import {
    Button,
    Typography,
    Paper,
    Box,
    CircularProgress,
    Alert,
} from "@mui/material";

const UploadForm = () => {
    const [file, setFile] = useState(null);
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();

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
        setLoading(true);
        const formData = new FormData();
        formData.append("file", file);
        try {
            const data = await uploadAudio(formData);
            setLoading(false);
            if (data.error) {
                setError(data.error);
            } else {
                // Navigate to results page with state (result data and file name)
                navigate("/results", { state: { result: data, fileName: file.name } });
            }
        } catch (err) {
            setLoading(false);
            setError("Failed to process audio.");
            console.error(err);
        }
    };

    return (
        <Paper elevation={3} sx={{ padding: 3, maxWidth: 500, margin: "auto", mt: 5 }}>
            <Typography variant="h5" gutterBottom>
                Upload Your Audio
            </Typography>
            <form onSubmit={handleUpload}>
                <Box display="flex" flexDirection="column" gap={2}>
                    <Button variant="outlined" component="label">
                        Choose File
                        <input type="file" hidden onChange={handleFileChange} accept=".mp3, .wav" />
                    </Button>
                    <Button variant="contained" color="primary" type="submit" disabled={loading}>
                        {loading ? <CircularProgress size={24} /> : "Upload & Analyze"}
                    </Button>
                </Box>
            </form>
            {error && (
                <Alert severity="error" sx={{ mt: 2 }}>
                    {error}
                </Alert>
            )}
        </Paper>
    );
};

export default UploadForm;
