#include "listset.h"
#include "listhack.h"

/*
 * TODO The last thing to try before giving up: output from meinolf's code the input/output to / from the list. Check if it matches here...
 */

long ListHack::operator[](int idx) {
    if(idx >= (int)sset->data.size()) {
        return -1;
    }
    return sset->data[idx];
//     if(idx >= (int)sset->data.size()) return -1;
//     auto itr = sset->data.begin();
//     std::cout << "wtf A: " << *itr << "; " << idx << " / " << sset->card << std::endl;
//     for(int ii = 0; ii < idx; ++ii) {
//         itr++; // this is obviously not good...
//         std::cout << "wtf A." << ii << "; " << *itr << std::endl;
//     }
//     std::cout << "wtf B: " << *itr << std::endl;
//     return *itr;
}
