# ui_config.py

custom_css = """
/* 1. Global Modern Reset */
footer {visibility: hidden}
.gradio-container {
    font-family: 'Inter', -apple-system, sans-serif !important;
    background: #f0f2f5;
    background-image: 
        radial-gradient(at 0% 0%, hsla(161,64%,80%,1) 0, transparent 50%), 
        radial-gradient(at 50% 0%, hsla(152,44%,85%,1) 0, transparent 50%), 
        radial-gradient(at 100% 0%, hsla(161,64%,80%,1) 0, transparent 50%);
    min-height: 100vh;
}

/* 2. The 'Floating Glass' Container */
#floating_container {
    background: rgba(255, 255, 255, 0.6);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.4);
    border-radius: 24px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
    width: 80%;
    max-width: 900px;
    margin: 40px auto;
    padding: 0; /* Remove padding to let child elements fit edge-to-edge */
    overflow: hidden;
}

/* 3. Sleek Modern Header */
.widget-header {
    background: transparent;
    color: #064e3b;
    padding: 24px;
    font-size: 1.5rem;
    font-weight: 700;
    text-align: left;
    border-bottom: 1px solid rgba(0, 0, 0, 0.05);
    display: flex;
    align-items: center;
    gap: 12px;
}

/* 4. Chat Interface Styling */
.gradio-container .chat-interface {
    background: transparent !important;
}

/* Modernizing the Input Box */
#component-7 { 
    border-radius: 16px !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    background: white !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03) !important;
}

/* Responsive Breakpoints */
@media (max-width: 768px) {
    #floating_container {
        width: 95%;
        margin: 10px auto;
    }
    .hero-section {
        padding: 20px !important;
    }
}
"""

hero_html = """
<div class="hero-section" style="padding: 60px 20px 20px 20px; text-align: center;">
    <h1 style="font-size: 3rem; font-weight: 800; color: #064e3b; letter-spacing: -1px; margin-bottom: 8px;">
        Farmer<span style="color: #10b981;">Bot</span>
    </h1>
    <p style="font-size: 1.1rem; color: #4b5563; max-width: 500px; margin: 0 auto 24px auto;">
        Your intelligent partner for sustainable and scientific farming decisions.
    </p>
    <div style="display: flex; justify-content: center; gap: 12px;">
        <div style="background: rgba(16, 185, 129, 0.1); color: #065f46; padding: 6px 16px; border-radius: 100px; font-weight: 600; font-size: 0.85rem; border: 1px solid rgba(16, 185, 129, 0.2);">
            ✓ Verified Handbook
        </div>
        <div style="background: rgba(0, 0, 0, 0.05); color: #374151; padding: 6px 16px; border-radius: 100px; font-weight: 600; font-size: 0.85rem;">
            ⚡ Gemini 2.5
        </div>
    </div>
</div>
"""