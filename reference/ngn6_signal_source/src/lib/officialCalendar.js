const CALENDAR_CACHE_MS = Number(process.env.CALENDAR_CACHE_MS || 6 * 60 * 60 * 1000);
const RELEVANT_EVENT_WINDOW_DAYS = Number(process.env.CALENDAR_EVENT_WINDOW_DAYS || 14);

const EVENT_SPECS = [
  {
    code: "EIA_GAS",
    label: "EIA weekly natural gas storage report",
    severity: "high",
    weekday: 4,
  },
  {
    code: "RIGS_GAS",
    label: "Baker Hughes gas rig count",
    severity: "scheduled",
    weekday: 5,
  },
];

let calendarCache = {
  value: null,
  expiresAt: 0,
  promise: null,
};

function classifyCalendarRisk(nearestEvent) {
  if (!nearestEvent) {
    return "none";
  }

  if (nearestEvent.daysUntil === 0 && nearestEvent.severity === "high") {
    return "high";
  }

  if (nearestEvent.daysUntil <= 1) {
    return "scheduled";
  }

  return "none";
}

function nextWeekdayDate(weekday, now = new Date()) {
  const today = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
  const currentWeekday = today.getUTCDay();
  const delta = (weekday - currentWeekday + 7) % 7;

  if (delta === 0) {
    return today;
  }

  const next = new Date(today);
  next.setUTCDate(next.getUTCDate() + delta);
  return next;
}

function getFallbackUpcomingEvent(spec, now = new Date()) {
  const date = nextWeekdayDate(spec.weekday, now);
  const start = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
  const daysUntil = Math.round((date.getTime() - start.getTime()) / 86_400_000);

  return {
    code: spec.code,
    label: spec.label,
    severity: spec.severity,
    date: date.toISOString().slice(0, 10),
    source: "gas-calendar-recurring",
    daysUntil,
  };
}

async function getMacroCalendar() {
  const now = Date.now();
  if (calendarCache.value && calendarCache.expiresAt > now) {
    return calendarCache.value;
  }

  if (calendarCache.promise) {
    return calendarCache.promise;
  }

  calendarCache.promise = (async () => {
    const events = EVENT_SPECS.map((spec) => getFallbackUpcomingEvent(spec))
      .filter((event) => event.daysUntil <= RELEVANT_EVENT_WINDOW_DAYS)
      .sort((left, right) => left.daysUntil - right.daysUntil);

    const nearest = events[0] || null;

    return {
      eventRisk: classifyCalendarRisk(nearest),
      events: events.slice(0, 5),
      source: "gas-calendar-recurring",
      warning: null,
    };
  })();

  try {
    const calendar = await calendarCache.promise;
    calendarCache = {
      value: calendar,
      expiresAt: Date.now() + CALENDAR_CACHE_MS,
      promise: null,
    };
    return calendar;
  } catch (error) {
    calendarCache = {
      value: null,
      expiresAt: 0,
      promise: null,
    };
    throw error;
  }
}

module.exports = {
  classifyCalendarRisk,
  getFallbackUpcomingEvent,
  getMacroCalendar,
};
