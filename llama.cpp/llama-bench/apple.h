typedef void* CFStringRef;
typedef void* CFDictionaryRef;
typedef void* CFMutableDictionaryRef;
typedef void* CFTypeRef;
typedef void* CFArrayRef;
typedef void* IOReportSubscriptionRef;
typedef void* CVoidRef;
typedef int CFIndex;

void init_apple_mon();

void am_release(void* obj);

void print_object(CFTypeRef obj);

CFMutableDictionaryRef get_power_channels();

IOReportSubscriptionRef get_subscription(CFMutableDictionaryRef power_channel);

CFDictionaryRef sample_power(IOReportSubscriptionRef sub, CFMutableDictionaryRef channels);

double sample_to_millijoules(CFDictionaryRef sample);