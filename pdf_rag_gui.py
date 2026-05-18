import customtkinter as ctk
import subprocess
import threading

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app=ctk.CTk()

app.geometry("1500x900")
app.title("PDF RAG STUDIO")

sidebar=ctk.CTkFrame(
    app,
    width=280,
    corner_radius=0
)

sidebar.pack(
    side="left",
    fill="y"
)

title=ctk.CTkLabel(
    sidebar,
    text="DOCUMENTOS",
    font=("Segoe UI",22,"bold")
)

title.pack(pady=20)

pdf_list=ctk.CTkTextbox(
    sidebar,
    width=250
)

pdf_list.pack(
    fill="both",
    expand=True,
    padx=10,
    pady=10
)

main=ctk.CTkFrame(app)

main.pack(
    side="right",
    fill="both",
    expand=True
)

answer_box=ctk.CTkTextbox(
    main,
    font=("Segoe UI",14)
)

answer_box.pack(
    fill="both",
    expand=True,
    padx=20,
    pady=20
)

bottom=ctk.CTkFrame(main)

bottom.pack(
    fill="x",
    padx=20,
    pady=10
)

question=ctk.CTkEntry(
    bottom,
    placeholder_text="Pregunta sobre tus PDFs..."
)

question.pack(
    side="left",
    fill="x",
    expand=True,
    padx=10
)

def ask():

    q=question.get()

    if not q:
        return

    answer_box.insert(
        "end",
        f"\n\nTÚ:\n{q}\n"
    )

    question.delete(0,"end")

    def worker():

        process=subprocess.run(
            [
                "PDF_RAG_LOCAL.exe",
                "ask",
                q
            ],
            capture_output=True,
            text=True
        )

        answer=process.stdout

        answer_box.insert(
            "end",
            f"\nIA:\n{answer}\n"
        )

    threading.Thread(
        target=worker
    ).start()

send=ctk.CTkButton(
    bottom,
    text="Enviar",
    command=ask
)

send.pack(
    side="right"
)

app.mainloop()
