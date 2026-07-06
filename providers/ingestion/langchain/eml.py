from __future__ import annotations
import email
from email import policy
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="langchain", name="eml")
class LangChainEMLLoader(BaseLoader):
    """Load email files (.eml) using stdlib — no extra packages needed."""

    @property
    def supported_types(self) -> list[str]:
        return [".eml"]

    def load(self, source: str) -> list[LoadedDocument]:
        with open(source, "rb") as f:
            msg = email.message_from_binary_file(f, policy=policy.default)

        subject = str(msg.get("subject", ""))
        sender  = str(msg.get("from", ""))
        date    = str(msg.get("date", ""))

        parts: list[str] = []
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        parts.append(payload.decode("utf-8", errors="replace"))
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                parts.append(payload.decode("utf-8", errors="replace"))

        content = f"Subject: {subject}\nFrom: {sender}\nDate: {date}\n\n" + "\n".join(parts)

        return [LoadedDocument(
            content=content,
            metadata={"subject": subject, "from": sender, "date": date, "source": source},
            source=source,
            provider="langchain",
            loader="eml",
        )]
