#include <iostream>

class IdCard
{
    private:
        std::string name;
        std::string dob;
        std::string addr;
    
    public:
        IdCard();
        IdCard(std::string, std::string, std::string);
        std::string get_name();
        std::string get_dob();
        std::string get_addr();
        void set_name(std::string);
        void set_addr(std::string);
        void set_dob(std::string);
        ~IdCard();
};