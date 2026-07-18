@echo off
start "LlamaCPP-Server" /MIN cmd /c "E:\Path\To\llama-server.exe -m E:\Software\MODELS_LLM\Qwen2.5-Coder-7B-Instruct-Q5_K_M.gguf -ngl 99 -c 12288 --port 8080"

timeout /t 5
