from zeep import Client

wsdl_path = r"C:C:\BLKQCL-SDK-4.0b14\Interfaces\BLKQCL-v2017-04.wsdl"
client = Client(wsdl=wsdl_path)
print(list(client.wsdl.services.keys()))
