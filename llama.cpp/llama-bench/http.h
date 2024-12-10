#include <string>
#include <map>

typedef std::map<std::string, std::string> Headers;

struct Response {
    int status;
    int content_length;
    std::string body;
};

Response GET(const std::string& url, const Headers& headers = {});

Response POST(const std::string& url, const std::string& body, const Headers& headers = {});