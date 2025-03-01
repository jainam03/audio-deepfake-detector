// src/components/ResultsPage.js
import React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Paper, Typography, Button, Box } from "@mui/material";

const ResultsPage = () => {
    const location = useLocation();
    const navigate = useNavigate();
    const { result, fileName } = location.state || {};

    // If no result data exists, redirect back to upload page
    if (!result) {
        navigate("/");
        return null;
    }

    return (
        <Paper elevation={3} sx={{ padding: 3, maxWidth: 500, margin: "auto", mt: 5 }}>
            <Typography variant="h5" gutterBottom>
                Analysis Results
            </Typography>
            <Box mt={2}>
                <Typography variant="subtitle1">File: {fileName}</Typography>
                <Typography variant="subtitle1">Result: {result.result_label}</Typography>
                <Typography variant="body1">
                    Fake Probability: {result.fake_probability.toFixed(2)}%
                </Typography>
                <Typography variant="body1">
                    Real Probability: {result.real_probability.toFixed(2)}%
                </Typography>
            </Box>
            <Button
                variant="outlined"
                color="primary"
                sx={{ mt: 3 }}
                onClick={() => navigate("/")}
            >
                Back to Upload
            </Button>
        </Paper>
    );
};

export default ResultsPage;
