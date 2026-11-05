import ipaddress
import socket
from datetime import datetime, timedelta, timezone


def lan_ip():
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("10.255.255.255", 1))
        return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


def ensure_certificate(folder, ip):
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError as error:
        raise SystemExit(
            "Phone and tablet mode needs the 'cryptography' package: pip install -r requirements-app.txt"
        ) from error

    folder.mkdir(parents=True, exist_ok=True)
    cert_path = folder / "cert.pem"
    key_path = folder / "key.pem"
    ip_path = folder / "ip.txt"
    if cert_path.exists() and key_path.exists() and ip_path.exists() and ip_path.read_text() == ip:
        return cert_path, key_path

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "PupilAD local server")])
    now = datetime.now(timezone.utc)
    alt_names = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    if ip != "127.0.0.1":
        alt_names.append(x509.IPAddress(ipaddress.ip_address(ip)))
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName(alt_names), critical=False)
        .sign(key, hashes.SHA256())
    )
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    ip_path.write_text(ip)
    return cert_path, key_path
