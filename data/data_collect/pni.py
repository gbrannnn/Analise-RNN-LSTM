import requests

class PNIDataCollector:
    def __init__(self, limit=100):
        self.base_url = "https://apidadosabertos.saude.gov.br/vacinacao/doses-aplicadas-pni-"
        self.limit = limit
        self.session = requests.Session()

    def collect_data(self, offset=0, year=2020, uf_estabelecimento="SP"):
        url = f"{self.base_url}{year}?uf_estabelecimento={uf_estabelecimento}&limit={self.limit}&offset={offset}"

        if self.session is None:
            self.session = requests.Session()
        
        response = self.session.get(url, headers={"Accept": "application/json"})
        return response
